"""Zero TERM grace delivers TERM without escalating an uncooperative namespace process."""

import os
import shlex
import subprocess
import uuid

import pytest

from tests.tools import test_docker_backend_real as real_docker
from tests.waiting import wait_until
from hermes_constants import get_hermes_home
from tools import terminal_tool
from tools.process_registry import process_registry

pytestmark = pytest.mark.skipif(not os.getenv("HERMES_TEST_DOCKER_IMAGE"),
                                reason="HERMES_TEST_DOCKER_IMAGE not set")
docker_task = real_docker.docker_task


@pytest.mark.platforms("linux")
def test_zero_grace_delivers_term_without_killing_survivors(docker_task):
    (get_hermes_home() / "config.yaml").write_text("terminal:\n  daemon_term_grace_seconds: 0\n")
    assert process_registry._daemon_term_grace_seconds() == 0
    real_docker._run("true", docker_task)
    env = terminal_tool._active_environments[real_docker._env_key(docker_task)]
    cid = real_docker._container_id(docker_task)
    root = f"/tmp/{uuid.uuid4().hex}"

    def control(command):
        return subprocess.run(["docker", "exec", cid, "bash", "-c", command],
                              capture_output=True, text=True, timeout=10, check=False)

    script = ("import os,pathlib,signal; "
              f"signal.signal(signal.SIGTERM,lambda *_:pathlib.Path({root + '.term'!r}).touch()); "
              f"pathlib.Path({root + '.pid'!r}).write_text(str(os.getpid())); "
              "exec('while True: signal.pause()')")
    proc = env._run_bash("exec python3 -c " + shlex.quote(script))
    pid = None
    try:
        wait_until(lambda: control(f"test -e {root}.pid").returncode == 0,
                   what="TERM handler installed in the namespace")
        pid = int(control(f"cat {root}.pid").stdout)
        env._kill_process(proc)
        alive = control(f"read -r s < /proc/{pid}/stat || exit 1; s=${{s##*) }}; "
                        'set -- $s; [ "$1" != Z ]')
        assert alive.returncode == 0, "zero grace escalated TERM to KILL"
        wait_until(lambda: control(f"test -e {root}.term").returncode == 0,
                   what="TERM delivered without escalation")
    finally:
        if pid is not None:
            control(f"kill -KILL {pid} 2>/dev/null || :")
        proc.wait(timeout=10)
        getattr(env, "_foreground_commands", {}).pop(id(proc), None)
