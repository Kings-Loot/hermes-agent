"""Run-local, bounded continuation after a folded user message."""

from dataclasses import dataclass, field
import threading

from agent.message_metadata import append_message

ASIDE_CONTEXT = (
    "This message arrived while you were working on the current task. Address it if needed, "
    "then continue the original task unless this message explicitly stops, cancels, or replaces it."
)
CONTINUATION_NOTE = (
    "Continue the original task from the current state, respecting the user's latest instructions. "
    "Do not repeat completed tool actions. If the original task is already complete, cancelled, "
    "or replaced and no work remains, reply NO_REPLY."
)


@dataclass
class AsideContinuity:
    pending: bool = False
    continued: bool = False
    delivered: set[str] = field(default_factory=set)
    lock: object = field(default_factory=threading.Lock)


def mark_aside(agent) -> None:
    state = getattr(agent, "_aside_continuity", None)
    if isinstance(state, AsideContinuity):
        with state.lock:
            state.pending = True


def cancel_asides(agent) -> None:
    state = getattr(agent, "_aside_continuity", None)
    if isinstance(state, AsideContinuity):
        with state.lock:
            state.pending = False


def aside_pending(agent) -> bool:
    state = getattr(agent, "_aside_continuity", None)
    return isinstance(state, AsideContinuity) and state.pending


def claim_continuation(agent) -> bool:
    state = getattr(agent, "_aside_continuity", None)
    if not isinstance(state, AsideContinuity):
        return False
    with state.lock:
        if not state.pending or getattr(agent, "_interrupt_requested", False):
            return False
        state.pending = False
        state.continued = True
        return True


def deliver_aside(agent, text: str) -> None:
    visible = agent._strip_think_blocks(text or "").strip()
    state = agent._aside_continuity
    if visible and visible != "NO_REPLY" and visible not in state.delivered:
        # Lifecycle delivery is independent of optional tool commentary/interim messages.
        if not agent._interim_content_fully_streamed(visible) and not agent._interim_text_was_delivered(visible):
            agent._emit_status(visible)
        state.delivered.add(visible)


def continue_after_aside(agent, final_msg, final_response, messages, conversation_history) -> bool:
    if not claim_continuation(agent):
        return False
    deliver_aside(agent, final_response)
    append_message(messages, final_msg)
    append_message(messages, {"role": "user", "content": CONTINUATION_NOTE})
    agent._session_messages = messages
    agent._persist_session(messages, conversation_history)
    return True


def suppress_duplicate_aside(agent, text: str) -> str:
    state = getattr(agent, "_aside_continuity", None)
    if isinstance(state, AsideContinuity) and text in state.delivered:
        return "NO_REPLY"
    return text
