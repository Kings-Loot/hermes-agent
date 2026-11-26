import asyncio
from types import SimpleNamespace

import pytest

from gateway import run


class OneTick:
    def __init__(self):
        self.done = False

    def is_set(self):
        return self.done

    def wait(self, timeout=None):
        self.done = True
        return True


def tick(monkeypatch, *, runner=None, loop=None):
    def heartbeat_only(label, fn):
        if label == "Runtime heartbeat":
            fn()

    monkeypatch.setattr(run, "_housekeeping_chore", heartbeat_only)
    monkeypatch.setattr("gateway.run_delivery_queue_watch.wait_for_next_tick", lambda stop, *_: stop.wait())
    run._start_gateway_housekeeping(OneTick(), runner=runner, loop=loop)


def test_housekeeping_refreshes_retired_work_on_the_gateway_loop(monkeypatch):
    state = {"active_agents": 1}

    async def scenario():
        loop = asyncio.get_running_loop()
        calls = []

        def liveness(**fields):
            calls.append("heartbeat")
            state.update(fields)

        def persist():
            assert asyncio.get_running_loop() is loop
            calls.append("count")
            state["active_agents"] = 0

        monkeypatch.setattr(run, "_write_runtime_status_quiet", liveness)
        await asyncio.to_thread(tick, monkeypatch, runner=SimpleNamespace(_persist_active_agents=persist), loop=loop)
        assert state["active_agents"] == 0
        assert calls == ["heartbeat", "count"]

    asyncio.run(scenario())


@pytest.mark.parametrize("missing", ["runner", "loop"])
def test_housekeeping_keeps_liveness_without_an_owner(monkeypatch, missing):
    calls = []
    monkeypatch.setattr(run, "_write_runtime_status_quiet", lambda **fields: calls.append(fields))
    runner = None if missing == "runner" else SimpleNamespace(_persist_active_agents=pytest.fail)

    async def scenario():
        loop = asyncio.get_running_loop() if missing == "runner" else None
        await asyncio.to_thread(tick, monkeypatch, runner=runner, loop=loop)

    asyncio.run(scenario())
    assert calls == [{}]


def test_housekeeping_keeps_liveness_if_loop_closes_during_scheduling(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "_write_runtime_status_quiet", lambda **fields: calls.append(fields))
    loop = asyncio.new_event_loop()
    schedule = loop.call_soon_threadsafe

    try:
        with monkeypatch.context() as race:
            race.setattr(loop, "is_running", lambda: True)

            def close_then_schedule(callback):
                race.undo()
                loop.close()
                return schedule(callback)

            race.setattr(loop, "call_soon_threadsafe", close_then_schedule)
            tick(monkeypatch, runner=SimpleNamespace(_persist_active_agents=pytest.fail), loop=loop)
        assert calls == [{}]
        assert loop.is_closed()
    finally:
        if not loop.is_closed():
            loop.close()
