"""Native app-server turns retain the same bounded aside contract."""

from unittest.mock import patch

import pytest

from agent.transports.codex_app_server_session import CodexAppServerSession, TurnResult
from tests.agent.test_codex_app_server_integration import _make_codex_agent


@pytest.mark.parametrize("mode", ["steer", "redirect"])
@pytest.mark.parametrize("answer", ["NO_REPLY", "The first work item is complete."])
def test_native_aside_continues_without_replaying_work(monkeypatch, mode, answer):
    inputs, corrections, delivered, writes = [], [], [], []
    agent = _make_codex_agent()
    agent.interim_assistant_callback = None
    agent.status_callback = lambda kind, message: delivered.append(message)
    monkeypatch.setattr(CodexAppServerSession, "ensure_started", lambda self: "thread-test")
    monkeypatch.setattr(CodexAppServerSession, "request_steer", lambda self, text: corrections.append(text) or True)

    def provider(self, user_input, **kwargs):
        inputs.append(user_input)
        item = len(inputs)
        writes.append(item)
        if item == 1:
            assert getattr(agent, mode)("How is the task going?")
        final = answer if item == 1 else "Finished both items."
        return TurnResult(final_text=final, projected_messages=[
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"w{item}", "type": "function",
             "function": {"name": "exec_command", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"w{item}", "content": f"Recorded {item}"},
            {"role": "assistant", "content": final},
        ], tool_iterations=1, interrupted=False, error=None, turn_id=f"t{item}", thread_id="thread-test")

    monkeypatch.setattr(CodexAppServerSession, "run_turn", provider)
    with patch.object(agent, "_cleanup_task_resources"):
        result = agent.run_conversation("Record both work items exactly once.")
    assert writes == [1, 2]
    assert result["api_calls"] == 2
    assert result["final_response"] == "Finished both items."
    assert result["aside_continued"] is True
    assert len(corrections) == 1
    assert "How is the task going?" in corrections[0]
    assert len([m for m in result["messages"] if m.get("role") == "tool"]) == 2
    assert delivered.count(answer) == (0 if answer == "NO_REPLY" else 1)
