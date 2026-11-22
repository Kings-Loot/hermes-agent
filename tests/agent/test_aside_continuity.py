"""A reply to mid-task input must not silently settle the original assignment."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from tests.agent.test_run_agent import _mock_response
from tools.registry import registry

TOOL = "aside_work_probe"
SCHEMA = {"name": TOOL, "description": "Record one work item", "parameters": {
    "type": "object", "properties": {"item": {"type": "integer"}}, "required": ["item"],
}}


def work(item):
    return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[SimpleNamespace(
        id=f"work_{item}", type="function", function=SimpleNamespace(name=TOOL, arguments=f'{{"item":{item}}}'),
    )])


def text(content):
    return _mock_response(content=content, finish_reason="stop")


@pytest.fixture
def harness():
    writes, delivered, payloads = [], [], []
    on_work = []

    def handler(args, **kwargs):
        writes.append(args["item"])
        if on_work:
            on_work.pop(0)()
        return "work recorded"

    registry.register(name=TOOL, toolset="utility", schema=SCHEMA, handler=handler, override=True)
    with (
        patch("model_tools.get_tool_definitions", return_value=[{"type": "function", "function": SCHEMA}]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=20)
    agent.client = MagicMock()
    agent._disable_streaming = True
    agent._cached_system_prompt = "Complete the user's task."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.interim_assistant_callback = None
    agent.status_callback = lambda kind, message: delivered.append(message)

    def run(script):
        steps = iter(script)

        def provider(kwargs):
            payloads.append(deepcopy(kwargs["messages"]))
            step = next(steps)
            return step() if callable(step) else step

        agent._interruptible_api_call = provider
        with (
            patch.object(agent, "_flush_messages_to_session_db"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            return agent.run_conversation("Record work items 1 and 2 exactly once.")

    def redirect(message="Devin, can you see the work?"):
        assert agent.redirect(message)
        raise InterruptedError("incoming user message")

    return SimpleNamespace(agent=agent, run=run, redirect=redirect, writes=writes,
                           delivered=delivered, payloads=payloads, on_work=on_work)


@pytest.mark.parametrize("mode", ["interrupt", "steer"])
@pytest.mark.parametrize("answer", ["NO_REPLY", "The first item is done; the second is pending."])
def test_aside_answer_continues_work_once(harness, mode, answer):
    h = harness
    if mode == "steer":
        h.on_work.append(lambda: h.agent.steer("How is the task going?"))
        script = [work(1), text(answer), work(2), text("Both items recorded.")]
    else:
        script = [work(1), h.redirect, text(answer), work(2), text("Both items recorded.")]
    result = h.run(script)
    assert h.writes == [1, 2]
    assert result["final_response"] == "Both items recorded."
    assert h.delivered.count(answer) == (0 if answer == "NO_REPLY" else 1)
    assert not result.get("response_previewed")
    assert all(p[0] == h.payloads[0][0] for p in h.payloads)
    first_tool = next(m for m in h.payloads[1] if m.get("role") == "tool")
    assert all(first_tool in p for p in h.payloads[1:])


def test_replacement_does_not_replay_completed_work(harness):
    h = harness
    h.on_work.append(lambda: h.agent.steer("Replace item 2 with item 3."))
    result = h.run([work(1), text("Switching to item 3."), work(3), text("Replacement finished.")])
    assert h.writes == [1, 3]
    assert result["final_response"] == "Replacement finished."
    assert h.delivered.count("Switching to item 3.") == 1


@pytest.mark.parametrize("reason", [None, "gateway shutdown"])
def test_stop_or_shutdown_does_not_resume_pending_aside(harness, reason):
    h = harness
    h.on_work.append(lambda: h.agent.steer("How is it going?"))

    def stop():
        h.agent.interrupt(hard_cancel=True, tool_reason=reason)
        raise InterruptedError("stop")

    result = h.run([work(1), stop])
    assert h.writes == [1]
    assert result["interrupted"]
    assert len(h.payloads) == 2


def test_each_new_aside_gets_only_one_continuation(harness):
    h = harness
    result = h.run([work(1), h.redirect, text("NO_REPLY"), h.redirect,
                    text("NO_REPLY"), text("NO_REPLY")])
    assert len(h.payloads) == 6
    assert result["final_response"] == "NO_REPLY"
    assert h.writes == [1]
    assert "NO_REPLY" not in h.delivered


@pytest.mark.parametrize("answer", ["NO_REPLY", "The task is complete."])
def test_completed_or_cancelled_task_can_remain_silent_after_one_check(harness, answer):
    h = harness
    h.on_work.append(lambda: h.agent.steer("Cancel the remaining work."))
    result = h.run([work(1), text(answer), text("NO_REPLY")])
    assert len(h.payloads) == 3
    assert h.writes == [1]
    assert result["final_response"] == "NO_REPLY"
    assert h.delivered.count(answer) == (0 if answer == "NO_REPLY" else 1)


def test_no_aside_has_no_extra_call(harness):
    h = harness
    result = h.run([work(1), text("NO_REPLY")])
    assert len(h.payloads) == 2
    assert result["final_response"] == "NO_REPLY"


def test_aside_state_does_not_leak_into_a_reused_agent(harness):
    h = harness
    h.on_work.append(lambda: h.agent.steer("How is it going?"))
    h.run([work(1), text("NO_REPLY"), text("NO_REPLY")])
    before = len(h.payloads)
    result = h.run([text("A new answer.")])
    assert len(h.payloads) == before + 1
    assert result["final_response"] == "A new answer."


def test_streamed_aside_is_not_sent_again_as_status(harness):
    h = harness
    h.on_work.append(lambda: h.agent.steer("How is it going?"))
    answer = "One item is recorded."
    h.agent._interim_content_fully_streamed = lambda content: content == answer
    result = h.run([work(1), text(answer), work(2), text("Finished.")])
    assert h.writes == [1, 2]
    assert answer not in h.delivered
    assert result["final_response"] == "Finished."


def test_repeated_final_does_not_duplicate_the_aside_answer(harness):
    h = harness
    h.on_work.append(lambda: h.agent.steer("Cancel the remaining work."))
    result = h.run([work(1), text("The remaining work is cancelled."), text("The remaining work is cancelled.")])
    assert h.writes == [1]
    assert h.delivered.count("The remaining work is cancelled.") == 1
    assert result["final_response"] == "NO_REPLY"
    assert result["aside_continued"] is True


def test_new_steer_after_continuation_never_inserts_into_the_cached_prefix(harness):
    from agent.turn_iteration_prep import _inject_steer_after_newest_tool_result

    messages = [{"role": "tool", "tool_call_id": "work1", "content": "recorded"},
                {"role": "user", "content": "first aside"},
                {"role": "assistant", "content": "NO_REPLY"},
                {"role": "user", "content": "continue original work"}]
    before = deepcopy(messages)
    _inject_steer_after_newest_tool_result(harness.agent, messages, "another question")
    assert messages[:len(before)] == before
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert "another question" in messages[-1]["content"]
