import asyncio
from types import SimpleNamespace
from threading import Event
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.platforms.event import MessageEvent, MessageType
from gateway.turn_context import TurnContext
from tests.gateway.test_authority_busy_preempt_parity import _admitted, _setup


def _model_message(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest_asyncio.fixture
async def active_turn(tmp_path, monkeypatch):
    runner, authority, source, agent, sent, turn = await _setup(tmp_path, monkeypatch, "interrupt")
    turn.event.text = "Run the database tests and report the results."
    turn.ctx = TurnContext()
    agent._supports_active_turn_redirect = True
    agent._hard_interrupt_requested = Event()
    agent._executing_tools = True
    agent.provider, agent.model, agent.base_url, agent.api_key, agent.api_mode = (
        "custom", "fixture", "http://127.0.0.1", "fixture", "chat_completions",
    )
    redirects = []

    def redirect(text):
        redirects.append(text)
        return True

    def hard_interrupt(*args, **kwargs):
        agent._hard_interrupt_requested.set()

    agent.redirect, agent.hard_interrupt = redirect, hard_interrupt
    try:
        yield runner, authority, source, agent, sent, turn, redirects
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,reply_expected,decision,answer,corrected",
    [
        ("Devin, check the release.", None, "[SILENT]", None, False),
        ("Devin, check the release.", False, None, None, False),
        ("Are you still running the tests?", None, '{"action":"answer","answer":"The task is still active."}',
         "The task is still active.", False),
        ("Correction: use the staging schema.", None, '{"action":"apply"}', None, True),
    ],
    ids=["unrelated", "adapter-marked-unrelated", "targeted-question", "targeted-correction"],
)
async def test_interjection_does_not_start_another_turn(
    active_turn, monkeypatch, text, reply_expected, decision, answer, corrected,
):
    runner, authority, source, agent, sent, turn, redirects = active_turn
    classify = AsyncMock(return_value=_model_message(decision))
    monkeypatch.setattr("gateway.busy_interjection.async_call_llm", classify)
    opener = turn.event
    ctx = turn.ctx
    key = runner._session_key_for_source(source)
    event = MessageEvent(text=text, source=source, message_id="followup", reply_expected=reply_expected)
    assert await asyncio.wait_for(runner._handle_message(event), 5) is None
    assert classify.await_count == (0 if reply_expected is False else 1)

    assert _admitted(authority) == []
    assert runner._peek_session_state(key).conversation.queued_events == []
    assert (turn.agent, turn.event, turn.ctx) == (agent, opener, ctx)
    assert agent.interrupts == []
    if corrected:
        assert len(redirects) == len(sent) == 1
        assert "continue the original task" in redirects[0]
        assert redirects[0].endswith(text)
    else:
        assert redirects == []
        assert opener.reply_anchor_override is None
        assert sent == ([answer] if answer else [])


@pytest.mark.asyncio
async def test_stop_while_interjection_decision_is_in_flight(active_turn, monkeypatch):
    runner, authority, source, agent, sent, _turn, redirects = active_turn
    started, release = asyncio.Event(), asyncio.Event()

    async def classify(**kwargs):
        started.set()
        await release.wait()
        return _model_message('{"action":"apply"}')

    monkeypatch.setattr("gateway.busy_interjection.async_call_llm", classify)
    event = MessageEvent(text="Use a different schema.", source=source, message_id="followup")
    interjection = asyncio.create_task(runner._handle_message(event))
    try:
        await asyncio.wait_for(started.wait(), 5)
        stop = MessageEvent(text="/stop", message_type=MessageType.COMMAND, source=source)
        await asyncio.wait_for(runner._handle_message(stop), 5)
        assert agent._hard_interrupt_requested.is_set()
        release.set()
        assert await asyncio.wait_for(interjection, 5) is None
        assert _admitted(authority) == []
        assert not redirects and not sent
    finally:
        release.set()
        await asyncio.gather(interjection, return_exceptions=True)
