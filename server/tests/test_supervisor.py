"""anvil.supervisor, with a stand-in trainer that crashes, finishes, fails or waits for Stop as told."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from anvil import supervisor

SERVER = Path(__file__).resolve().parents[1]

# Each launch does the next thing on the plan (the last one again once the plan runs out): it adds 10 steps, and
# "crash, saved" saves a checkpoint there first. A crash is SIGKILL, which ends it as a segfault would, without a core.
TRAINER = """
import json, os, signal, sys, time
from pathlib import Path

run = Path(sys.argv[1])
launches = json.loads((run / "launches.json").read_text()) if (run / "launches.json").exists() else []
launches.append(sys.argv[2:])
(run / "launches.json").write_text(json.dumps(launches))
plan = json.loads((run / "plan.json").read_text())
act = plan[min(len(launches), len(plan)) - 1]
status = json.loads((run / "status.json").read_text())
status.update(step=status.get("step", 0) + 10, pid=os.getpid(),
              state=act if act in ("done", "failed", "stopped") else "running")
if act == "crash, saved":
    folder = run / "checkpoints" / f"step-{status['step']:07d}"
    folder.mkdir(parents=True)
    (folder / "state.json").write_text("{}")
(run / "status.json").write_text(json.dumps(status))
if act.startswith("crash"):
    os.kill(os.getpid(), signal.SIGKILL)
if act == "wait":
    time.sleep(60)
"""


def supervise(run: Path, plan: list[str]) -> subprocess.Popen:
    run.mkdir(parents=True)
    (run / "trainer.py").write_text(TRAINER)
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "status.json").write_text(json.dumps({"state": "starting", "step": 0}))
    log = (run / "log.txt").open("ab")
    return subprocess.Popen(
        [sys.executable, "-m", "anvil.supervisor", str(run), "--", sys.executable, str(run / "trainer.py"), str(run)],
        env={**os.environ, "PYTHONPATH": str(SERVER)}, stdout=log, stderr=subprocess.STDOUT)


def status(run: Path) -> dict:
    return json.loads((run / "status.json").read_text())


def launches(run: Path) -> list[list[str]]:
    return json.loads((run / "launches.json").read_text())


def test_a_crash_starts_the_trainer_again_with_resume(tmp_path):
    run = tmp_path / "run"
    assert supervise(run, ["crash", "done"]).wait(timeout=60) == 0
    st = status(run)
    assert st["state"] == "done" and st["crashes"] == 1
    assert st["last_crash"]["step"] == 10 and st["last_crash"]["from"] is None
    assert launches(run) == [[], ["--resume"]]
    assert "(killed by SIGKILL): restarting from the start, try 1 of 3" in (run / "log.txt").read_text()


def test_crashing_again_and_again_from_one_checkpoint_gives_up(tmp_path):
    run = tmp_path / "run"
    assert supervise(run, ["crash, saved", "crash"]).wait(timeout=60) == 0
    st = status(run)
    # Every crash leaves step 10 as the newest checkpoint: three restarts from it, and the fourth crash gives up.
    assert len(launches(run)) == supervisor.MAX_TRIES + 1
    assert st["state"] == "failed" and st["crashes"] == 4 and st["last_crash"]["from"] == 10
    assert st["error"].startswith("The trainer crashed 4 times in a row from the step-10 checkpoint")


def test_a_newer_checkpoint_counts_as_progress(tmp_path):
    run = tmp_path / "run"
    assert supervise(run, ["crash, saved"] * 5 + ["done"]).wait(timeout=60) == 0
    st = status(run)
    assert st["state"] == "done" and st["crashes"] == 5 and st["last_crash"]["from"] == 50
    assert len(launches(run)) == 6


def test_an_end_the_trainer_reports_is_left_alone(tmp_path):
    for end in ("done", "failed", "stopped"):
        run = tmp_path / end
        assert supervise(run, [end]).wait(timeout=60) == 0
        st = status(run)
        assert st["state"] == end and "crashes" not in st and len(launches(run)) == 1


def test_stop_reaches_the_trainer_through_the_supervisor_and_nothing_starts_again(tmp_path):
    run = tmp_path / "run"
    proc = supervise(run, ["wait"])
    deadline = time.time() + 30
    while time.time() < deadline and status(run).get("state") != "running":
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=60) == 0
    st = status(run)
    assert st["state"] == "stopped" and "crashes" not in st and len(launches(run)) == 1
