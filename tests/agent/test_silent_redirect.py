from unittest.mock import patch

from tests.agent import test_run_agent

agent = test_run_agent.agent
_mock_plugin_discovery = test_run_agent._mock_plugin_discovery


def test_silent_midturn_input_does_not_end_original_work(agent):
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    requests = []

    def respond(api_kwargs):
        requests.append(api_kwargs["messages"])
        if len(requests) == 1:
            assert agent.redirect("Devin, please check the staging deployment.")
            raise InterruptedError("redirect cancelled the first request")
        if len(requests) == 2:
            return test_run_agent._mock_response(content="[SILENT]", finish_reason="stop")
        return test_run_agent._mock_response(content="Original report completed.", finish_reason="stop")

    with patch.object(agent, "_interruptible_api_call", side_effect=respond):
        result = agent.run_conversation("Complete the original report.")

    assert result["final_response"] == "Original report completed."
    assert len(requests) == 3
    assert sum(m.get("content") == "Complete the original report." for m in requests[-1]) == 1
    assert requests[-1][-1]["content"] == "Continue the original task from the progress already made."
