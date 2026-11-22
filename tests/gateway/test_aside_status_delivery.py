"""Mandatory mid-turn replies use lifecycle delivery, not optional commentary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.run import GatewayRunner
from gateway.config import GatewayConfig
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source


@pytest.mark.parametrize("platform", [Platform.SLACK, Platform.TELEGRAM])
def test_status_reply_with_interim_disabled_does_not_seal_final_stream(platform):
    adapter = SimpleNamespace(send_or_update_status=AsyncMock())
    metadata = {"thread_id": "thread-1"}
    progress = GatewayTurnProgressMixin()
    progress._ctx = SimpleNamespace(
        mute_notification_reply=False, source=SimpleNamespace(platform=platform),
        user_config={"display": {"interim_assistant_messages": False}},
        _status_adapter=adapter, _status_chat_id="chat-1", _status_thread_metadata=metadata,
        _cleanup_progress=False, _run_still_current=lambda: True,
    )
    progress._schedule = lambda coroutine, label: asyncio.run(coroutine)
    progress._status_callback_sync("lifecycle", "The first item is finished.")
    adapter.send_or_update_status.assert_awaited_once_with(
        "chat-1", "lifecycle", "The first item is finished.",
        metadata={"thread_id": "thread-1", "_interim_send": True},
    )
    assert metadata == {"thread_id": "thread-1"}


@pytest.mark.asyncio
async def test_internal_aside_continuation_may_end_silently_on_an_addressed_turn(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "NO_REPLY", "aside_continued": True,
        "messages": [{"role": "user", "content": "original assignment"},
                     {"role": "assistant", "content": "NO_REPLY"}],
        "tools": [], "history_offset": 0, "last_prompt_tokens": 0, "api_calls": 3, "failed": False,
    })
    response = await runner._handle_message_with_agent(
        _event(reply_expected=True), _source(), "agent:main:telegram:group:-1001:12345", 1,
    )
    assert response == ""


@pytest.mark.asyncio
async def test_silent_aside_continuation_is_also_silent_before_a_queued_followup():
    runner = GatewayRunner(GatewayConfig())
    runner._deliver_queued_first_response = AsyncMock()
    ctx = SimpleNamespace(
        session_key="agent:main:telegram:group:-1001:12345", stream_consumer_holder=[None],
        mute_notification_reply=False, persist_user_display_kind=None, reply_expected=True,
        source=_source(), _status_thread_metadata=None, event_message_id=None,
        inbound_message_id="msg-42", run_generation=1,
    )
    result = {"final_response": "NO_REPLY", "failed": False, "aside_continued": True}
    await runner._run_agent_deliver_first_response(ctx, None, result, result, None)
    runner._deliver_queued_first_response.assert_not_awaited()
