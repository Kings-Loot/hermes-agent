import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.response_filters import is_intentional_silence_response
from gateway.run_turn_runner import TurnRunner
from tests.agent import test_run_agent

agent = test_run_agent.agent
_mock_plugin_discovery = test_run_agent._mock_plugin_discovery


@pytest.fixture
def slack_turn():
    adapter = SimpleNamespace(send=AsyncMock())
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.SLACK, chat_id="chat"),
        user_config={"display": {"interim_assistant_messages": False}},
        mute_notification_reply=False, scheduled_heartbeat=False,
        streaming_tts_consumer_holder=[None], stream_consumer_holder=[None],
        interim_assistant_messages_enabled=False, resolve_display_setting=lambda *a: False,
        _run_still_current=lambda: True, _status_adapter=adapter, _status_chat_id="chat",
        _status_thread_metadata={"thread_id": "thread"},
        progress_callback=None, _voice_ack_guild=[None], _native_slack_task_cards=False,
        _hooks_ref=SimpleNamespace(loaded_hooks=[]), _status_callback_sync=None,
        _event_callback_sync=None, session_key="", voice_turn=False, _thinking_enabled=False,
        agent_holder=[None], tools_holder=[None], process_task_id=None, process_baseline=None,
    )
    turn._runner = SimpleNamespace(
        config=None, _service_tier=None, _delivery_adapter_for=lambda source: adapter,
        _consume_pending_turn_sidecar_notes=lambda key: [],
    )
    turn._approval_owner = None
    return turn, adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["question", "correction", "finished", "stop"])
async def test_standalone_redirect_answer_reaches_slack_before_original_task_finishes(agent, slack_turn, case):
    turn, adapter = slack_turn
    turn._ctx._loop_for_step = asyncio.get_running_loop()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    _, delta, interim, want_interim = turn._setup_stream_consumer("slack")
    turn._wire_turn_agent_callbacks(agent, {}, None, delta, interim, want_interim)

    followup = "Use staging, not production." if case == "correction" else "Which schema are you using?"
    answer = "The report is already complete." if case == "finished" else "I am using staging."
    answer_sent = threading.Event()
    def send(*args, **kwargs):
        answer_sent.set()
        return SimpleNamespace(success=True)

    adapter.send.side_effect = send
    tool_calls = []
    requests = []

    def respond(api_kwargs):
        messages = api_kwargs["messages"]
        requests.append(messages)
        if len(requests) == 1:
            assert agent.redirect(followup)
            raise InterruptedError("redirect")
        if len(requests) == 2:
            assert followup in messages[-1]["content"]
            reply = test_run_agent._mock_response(content=answer)
            if case == "question":
                reply.choices[0].message.codex_message_items = [
                    {"type": "message", "phase": "commentary",
                     "content": [{"type": "output_text", "text": "Internal working note."}]},
                ]
            return reply
        assert answer_sent.wait(timeout=5)
        assert [row["content"] for row in messages if row["role"] == "assistant"].count(answer) == 1
        if case == "stop":
            agent.hard_interrupt()
            raise InterruptedError("stop")
        if case == "finished":
            assert "[SILENT]" in messages[-1]["content"]
            return test_run_agent._mock_response(content="[SILENT]")
        if len(requests) == 3:
            args = '{"query":"staging"}' if case == "correction" else '{"query":"original report"}'
            tc = test_run_agent._mock_tool_call(name="web_search", arguments=args, call_id="original-tool")
            return test_run_agent._mock_response(content="Internal working note.", finish_reason="tool_calls", tool_calls=[tc])
        assert tool_calls
        return test_run_agent._mock_response(content="Original report completed.")

    def execute(_name, arguments, *args, **kwargs):
        tool_calls.append(arguments["query"])
        return "original tool result"

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=respond),
        patch("model_tools.handle_function_call", side_effect=execute),
    ):
        result = await asyncio.to_thread(agent.run_conversation, "Complete the original report.")

    assert [call.args[1] for call in adapter.send.await_args_list] == [answer]
    if case == "stop":
        assert result["interrupted"]
    elif case == "finished":
        assert is_intentional_silence_response(result["final_response"])
        runner = object.__new__(GatewayRunner)
        _, silence, _ = await runner._hmwa_shape_agent_response(
            result, turn._ctx.source, [], SimpleNamespace(session_id=agent.session_id),
            None, None, None, agent.session_id, "slack", time.time(), reply_expected=True,
        )
        assert silence
    else:
        assert result["final_response"] == "Original report completed."
    assert len(requests) == (3 if case in {"finished", "stop"} else 4)
    assert tool_calls == {"question": ["original report"], "correction": ["staging"]}.get(case, [])
