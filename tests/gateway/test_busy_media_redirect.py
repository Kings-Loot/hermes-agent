"""Busy media retains its attachment context without cancelling foreground work."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["busy", "priority"])
@pytest.mark.parametrize("kind,caption,mime,transcript", [
    (MessageType.DOCUMENT, "Read this too", "text/plain", None),
    (MessageType.PHOTO, "", "image/png", None),
    (MessageType.TEXT, "See attachment", "application/pdf", None),
    (MessageType.VOICE, "", "audio/ogg", "voice correction"),
    (MessageType.VOICE, "Use the cached voice", "audio/ogg", None),
])
async def test_interrupt_media_redirects_without_interrupting(route, kind, caption, mime, transcript, monkeypatch):
    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(platform=Platform.SLACK, chat_id="lab", user_id="requester", thread_id="thread")
    event = MessageEvent(text=caption, source=source, message_type=kind,
                         media_urls=["/cached/attachment"], media_types=[mime], message_id="followup")
    received = []

    class Agent:
        _supports_active_turn_redirect = True

        def redirect(self, text):
            received.append(text)
            return True

        def interrupt(self, text):
            pytest.fail("an attachment must not cancel the active command")

    agent = Agent()
    runner._session_state("key").turn.agent = agent
    opener = MessageEvent(text="original", source=source)
    runner._session_state("key").turn.event = opener
    monkeypatch.setattr(runner, "_transcribe_and_echo_pending_voice",
                        AsyncMock(return_value=(transcript or "", [transcript] if transcript else [])))
    if route == "priority":
        await runner._hm_busy_interrupt(event, source, agent, "key")
    else:
        outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
        assert outcome.redirected
    assert len(received) == 1
    assert (transcript or "/cached/attachment") in received[0]
    assert caption in received[0]
    assert opener.media_urls == event.media_urls
    assert opener.media_types == event.media_types


@pytest.mark.asyncio
async def test_command_is_not_reinterpreted_as_a_media_redirect():
    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(platform=Platform.SLACK, chat_id="lab", user_id="requester")
    event = MessageEvent(text="/stop", source=source, message_type=MessageType.COMMAND,
                         media_urls=["/cached/attachment"], media_types=["text/plain"])

    class Agent:
        _supports_active_turn_redirect = True

        def redirect(self, text):
            pytest.fail("commands retain their control semantics")

    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", Agent())
    assert not outcome.redirected


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "plain correction"])
async def test_redirect_without_attachments_keeps_text_contract(text):
    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(platform=Platform.SLACK, chat_id="lab", user_id="requester")
    event = MessageEvent(text=text, source=source)
    received = []

    class Agent:
        _supports_active_turn_redirect = True

        def redirect(self, value):
            received.append(value)
            return True

    agent = Agent()
    runner._session_state("key").turn.agent = agent
    outcome = await runner._resolve_busy_steer_or_redirect(event, "key", "interrupt", agent)
    assert outcome.redirected
    assert len(received) == 1 and text in received[0]
    assert not event.media_urls


@pytest.mark.asyncio
async def test_failed_media_redirect_retains_legacy_interrupt_fallback():
    runner = GatewayRunner(GatewayConfig())
    source = SessionSource(platform=Platform.SLACK, chat_id="lab", user_id="requester")
    event = MessageEvent(text="correction", source=source, message_type=MessageType.DOCUMENT,
                         media_urls=["/cached/note.txt"], media_types=["text/plain"])
    received = []

    class Agent:
        _supports_active_turn_redirect = True

        def redirect(self, text):
            raise RuntimeError("redirect unavailable")

        def interrupt(self, text):
            received.append(text)

    agent = Agent()
    runner._session_state("key").turn.agent = agent
    await runner._hm_busy_interrupt(event, source, agent, "key")
    assert received == ["correction"]
    assert event.media_urls == ["/cached/note.txt"]
