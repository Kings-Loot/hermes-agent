"""Publish gateway liveness and refresh its active-agent count."""


def write_housekeeping_heartbeat(write_status, runner, loop):
    write_status()
    if runner is None or loop is None or not loop.is_running():
        return
    try:
        loop.call_soon_threadsafe(runner._persist_active_agents)
    except RuntimeError:
        if not loop.is_closed():
            raise
