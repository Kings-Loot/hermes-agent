from unittest.mock import patch

import pytest

from tests.agent import test_run_agent

agent = test_run_agent.agent
_mock_plugin_discovery = test_run_agent._mock_plugin_discovery


@pytest.fixture(autouse=True)
def configure_agent(agent):
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False


@pytest.mark.parametrize("final", ["Original report completed.", "[SILENT]"])
def test_silent_midturn_input_does_not_end_original_work(agent, final):
    requests = []

    def respond(api_kwargs):
        requests.append(api_kwargs["messages"])
        if len(requests) == 1:
            assert agent.redirect("Devin, please check the staging deployment.")
            raise InterruptedError("redirect cancelled the first request")
        if len(requests) == 2:
            return test_run_agent._mock_response(content="[SILENT]", finish_reason="stop")
        return test_run_agent._mock_response(content=final, finish_reason="stop")

    with patch.object(agent, "_interruptible_api_call", side_effect=respond):
        result = agent.run_conversation("Complete the original report.")

    assert len(requests) == 3
    assert result["final_response"] == final
    assert sum(m.get("content") == "Complete the original report." for m in requests[-1]) == 1
    assert requests[-1][-1]["content"].startswith("Continue the original task from the progress already made.")


@pytest.mark.parametrize("followups", [1, 2])
def test_tool_redirect_keeps_result_and_continues_once(agent, followups):
    requests = []
    tc = test_run_agent._mock_tool_call(name="web_search", arguments="{}", call_id="original-tool")

    def execute(*args, **kwargs):
        assert agent._executing_tools
        for index in range(followups):
            assert agent.redirect(f"Devin, check deployment {index}.")
        return "original tool result"

    def respond(api_kwargs):
        messages = api_kwargs["messages"]
        requests.append(messages)
        if len(requests) == 1:
            return test_run_agent._mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        assert [row["content"] for row in messages if row["role"] == "tool"] == ["original tool result"]
        return test_run_agent._mock_response(content="[SILENT]" if len(requests) == 2 else "Original report completed.")

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=respond),
        patch("model_tools.handle_function_call", side_effect=execute) as tool,
    ):
        result = agent.run_conversation("Complete the original report.")

    tool.assert_called_once()
    assert len(requests) == 3
    assert result["final_response"] == "Original report completed."
    for index in range(followups):
        assert sum(f"check deployment {index}" in row["content"] for row in requests[-1]
                   if isinstance(row.get("content"), str)) == 1


@pytest.mark.parametrize("stop_request", [2, 3])
def test_stop_during_redirect_or_continuation(agent, stop_request):
    calls = 0

    def respond(_api_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert agent.redirect("Devin, check the staging deployment.")
            raise InterruptedError("redirect")
        if calls == stop_request:
            agent.hard_interrupt()
            raise InterruptedError("stop")
        return test_run_agent._mock_response(content="[SILENT]")

    with patch.object(agent, "_interruptible_api_call", side_effect=respond):
        result = agent.run_conversation("Complete the original report.")

    assert calls == stop_request
    assert result["interrupted"]


def test_targeted_redirect_answers_interim_then_finishes_original_work(agent):
    requests = []
    answer = "Yes, I will use the staging schema."
    commentary = []
    agent.interim_assistant_callback = lambda text, **kwargs: commentary.append(text)
    tc = test_run_agent._mock_tool_call(name="web_search", arguments="{}", call_id="original-tool")

    def respond(api_kwargs):
        requests.append(api_kwargs["messages"])
        if len(requests) == 1:
            assert agent.redirect("Can you use the staging schema?")
            raise InterruptedError("redirect")
        if len(requests) == 2:
            assert requests[-1][-1]["content"].endswith("Can you use the staging schema?")
            return test_run_agent._mock_response(content=answer, finish_reason="tool_calls", tool_calls=[tc])
        return test_run_agent._mock_response(content="Original report completed.")

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=respond),
        patch("model_tools.handle_function_call", return_value="original tool result") as tool,
    ):
        result = agent.run_conversation("Complete the original report.")

    tool.assert_called_once()
    assert commentary == [answer]
    assert len(requests) == 3
    assert result["final_response"] == "Original report completed."


def test_streamed_redirect_reply_resumes_without_interim_callback(agent):
    answer = "Yes, I will use the staging schema."
    rendered = []
    requests = []
    agent.stream_delta_callback = rendered.append

    def respond(api_kwargs, **_):
        requests.append(api_kwargs["messages"])
        if len(requests) == 1:
            assert agent.redirect("Can you use the staging schema?")
            raise InterruptedError("redirect")
        if len(requests) == 2:
            agent._fire_stream_delta(answer)
            return test_run_agent._mock_response(content=answer)
        return test_run_agent._mock_response(content="Original report completed.")

    with patch.object(agent, "_interruptible_streaming_api_call", side_effect=respond):
        result = agent.run_conversation("Complete the original report.")

    assert "".join(rendered).strip() == answer
    assert len(requests) == 3
    assert result["final_response"] == "Original report completed."
