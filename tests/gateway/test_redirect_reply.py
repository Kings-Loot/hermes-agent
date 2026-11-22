import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.run import GatewayRunner
from gateway.response_filters import is_intentional_silence_response
from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import StreamConsumerConfig
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
@pytest.mark.parametrize("case", ["question", "question-with-tool", "correction", "finished", "stop"])
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
    answered_before_tool = []
    requests = []

    def respond(api_kwargs):
        messages = api_kwargs["messages"]
        requests.append(messages)
        if len(requests) == 1:
            assert agent.redirect(followup)
            raise InterruptedError("redirect")
        if len(requests) == 2:
            assert followup in messages[-1]["content"]
            if case == "question-with-tool":
                tc = test_run_agent._mock_tool_call(
                    name="web_search", arguments='{"query":"original report"}', call_id="original-tool",
                )
                return test_run_agent._mock_response(content=answer, finish_reason="tool_calls", tool_calls=[tc])
            reply = test_run_agent._mock_response(content=answer)
            if case == "question":
                reply.choices[0].message.codex_message_items = [
                    {"type": "message", "phase": "commentary",
                     "content": [{"type": "output_text", "text": "Internal working note."}]},
                ]
            return reply
        if case != "question-with-tool":
            assert answer_sent.wait(timeout=5)
        assert [row["content"] for row in messages if row["role"] == "assistant"].count(answer) == 1
        if case == "stop":
            agent.hard_interrupt()
            raise InterruptedError("stop")
        if case == "finished":
            assert "[SILENT]" in messages[-1]["content"]
            return test_run_agent._mock_response(content="[SILENT]")
        if len(requests) == 3 and case != "question-with-tool":
            args = '{"query":"staging"}' if case == "correction" else '{"query":"original report"}'
            tc = test_run_agent._mock_tool_call(name="web_search", arguments=args, call_id="original-tool")
            return test_run_agent._mock_response(content="Internal working note.", finish_reason="tool_calls", tool_calls=[tc])
        assert tool_calls
        return test_run_agent._mock_response(content="Original report completed.")

    def execute(_name, arguments, *args, **kwargs):
        answered_before_tool.append(answer_sent.is_set())
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
    assert len(requests) == (3 if case in {"finished", "stop", "question-with-tool"} else 4)
    assert tool_calls == {
        "question": ["original report"], "question-with-tool": ["original report"], "correction": ["staging"],
    }.get(case, [])
    if case == "question-with-tool":
        assert answered_before_tool == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("silent", [True, False])
async def test_streamed_redirect_delivers_without_leaking_into_resumed_tool_preamble(agent, slack_turn, silent):
    turn, adapter = slack_turn
    turn._runner.config = SimpleNamespace(streaming=StreamingConfig(enabled=True))
    turn._runner._build_stream_consumer_config = lambda *a, **kw: (StreamConsumerConfig(buffer_only=True), None)
    turn._ctx.resolve_display_setting = lambda *a: None
    turn._ctx._loop_for_step = asyncio.get_running_loop()
    turn._ctx.progress_queue = None
    turn._ctx.event_message_id = None
    adapter.send.return_value = SimpleNamespace(success=True, message_id="message")
    adapter.edit_message = AsyncMock(return_value=adapter.send.return_value)
    agent._cached_system_prompt = "You are helpful."
    agent.compression_enabled = False
    agent.save_trajectories = False
    consumer, delta, interim, want_interim = turn._setup_stream_consumer("slack")
    turn._wire_turn_agent_callbacks(agent, {}, None, delta, interim, want_interim)
    calls = []
    tc = test_run_agent._mock_tool_call(name="web_search", arguments="{}", call_id="original-tool")

    def respond(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            assert agent.redirect("Devin, check deployment.")
            raise InterruptedError("redirect")
        if len(calls) == 2:
            first = "[SILENT]" if silent else "I am using staging."
            agent._fire_stream_delta(first)
            return test_run_agent._mock_response(content=first, finish_reason="stop" if silent else "length")
        if not silent and len(calls) == 3:
            last = "It has the expected tables."
            agent._fire_stream_delta(last)
            return test_run_agent._mock_response(content=last)
        if len(calls) == (3 if silent else 4):
            agent._fire_stream_delta("Checking the report.")
            return test_run_agent._mock_response(
                content="Checking the report.", finish_reason="tool_calls", tool_calls=[tc],
            )
        return test_run_agent._mock_response(content="Original report completed.")

    stream_task = asyncio.create_task(consumer.run())
    try:
        with (
            patch.object(agent, "_interruptible_streaming_api_call", side_effect=respond),
            patch("model_tools.handle_function_call", return_value="original tool result") as tool,
        ):
            result = await asyncio.to_thread(agent.run_conversation, "Complete the original report.")
        tool.assert_called_once()
    finally:
        consumer.finish()
        await asyncio.wait_for(stream_task, timeout=5)

    assert result["final_response"] == "Original report completed."
    adapter.send.assert_awaited()
    payloads = repr([*adapter.send.await_args_list, *adapter.edit_message.await_args_list])
    assert "Checking the report." in payloads
    assert "[SILENT]" not in payloads
    if not silent:
        assert payloads.count("I am using staging.") == 1
        assert payloads.count("It has the expected tables.") == 1
