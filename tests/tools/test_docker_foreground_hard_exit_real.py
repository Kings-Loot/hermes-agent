"""A hard exit dispatches all Docker namespace kills under one shared budget."""

import os
import shlex
import subprocess
import threading
import uuid

import pytest

from agent import secret_scope
from tests.tools import test_docker_backend_real as real_docker
from tests.waiting import wait_until
from tools import terminal_tool
from tools.environments import base

pytestmark = pytest.mark.skipif(not os.getenv("HERMES_TEST_DOCKER_IMAGE"),
                                reason="HERMES_TEST_DOCKER_IMAGE not set")
docker_task = real_docker.docker_task


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("multiplex", [False, True])
def test_hard_exit_dispatches_immediate_group_kills_without_serial_waits(docker_task, monkeypatch, multiplex, caplog):
    real_docker._run("true", docker_task)
    env = terminal_tool._active_environments[real_docker._env_key(docker_task)]
    cid = real_docker._container_id(docker_task)
    was_multiplex = secret_scope._MULTIPLEX_ACTIVE

    def probe(command):
        return subprocess.run(["docker", "exec", cid, "bash", "-c", command],
                              capture_output=True, text=True, timeout=10, check=False)

    release, returned = threading.Event(), threading.Event()
    entered, commands = set(), []
    errors = []
    shutdown = None
    root = f"/tmp/{uuid.uuid4().hex}"
    original_force = env._force_kill_process

    def blocked(proc):
        entered.add(proc.pid)
        if not multiplex:
            assert release.wait(20), "test did not release namespace control"
        original_force(proc)

    try:
        for index in range(3):
            script = ("import os,pathlib,signal; "
                      f"signal.signal(signal.SIGTERM,lambda *_:pathlib.Path({root + str(index) + '.term'!r}).touch()); "
                      f"pathlib.Path({root + str(index) + '.pid'!r}).write_text(str(os.getpid())); "
                      "exec('while True: signal.pause()')")
            proc = env._run_bash("exec python3 -c " + shlex.quote(script))
            commands.append((proc, None))
            wait_until(lambda: probe(f"test -e {root}{index}.pid").returncode == 0,
                       what="TERM-resistant foreground group ready")
            commands[-1] = (proc, int(probe(f"cat {root}{index}.pid").stdout))

        monkeypatch.setattr(env, "_force_kill_process", blocked)
        monkeypatch.setattr(base, "_live_foreground", {id(proc): (env, proc) for proc, _ in commands})
        monkeypatch.setattr(base, "_spawns_in_flight", 0)
        monkeypatch.setattr(base, "_exit_fenced", False)
        if multiplex:
            monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
            caplog.set_level("DEBUG", logger="tools.environments.base")

        def hard_exit():
            token = secret_scope.set_secret_scope({"HERMES_RB_FORWARDED": "routed-profile-value"})
            try:
                assert base.kill_live_foreground_processes(now=True) == len(commands)
            except BaseException as exc:
                errors.append(exc)
            finally:
                secret_scope.reset_secret_scope(token)
                returned.set()

        shutdown = threading.Thread(target=hard_exit, daemon=True)
        shutdown.start()
        assert returned.wait(2), "hard shutdown waited for a Docker control round trip"
        assert not errors, errors
        assert entered == {proc.pid for proc, _ in commands}, "Docker force kills serialized"
        release.set()
        for proc, pid in commands:
            wait_until(lambda: proc.poll() is not None, what="force-killed exec client exits")
            wait_until(lambda: probe(f"test ! -e /proc/{pid}/stat").returncode == 0,
                       what="namespace group killed without TERM grace or workload secrets")
        for index in range(len(commands)):
            assert probe(f"test ! -e {root}{index}.term").returncode == 0
    finally:
        release.set()
        if shutdown is not None:
            shutdown.join(timeout=15)
        if multiplex:
            monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", was_multiplex)
        for proc, pid in commands:
            if pid is not None:
                probe(f"kill -KILL {pid} 2>/dev/null || :")
            proc.wait(timeout=10)
            getattr(env, "_foreground_commands", {}).pop(id(proc), None)
