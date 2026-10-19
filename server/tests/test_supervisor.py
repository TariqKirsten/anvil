"""anvil.supervisor, with a stand-in trainer that crashes, finishes, fails, waits for Stop or can't stop, as told."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from anvil import supervisor

SERVER = Path(__file__).resolve().parents[1]

# Each launch does the next thing on the plan (the last one again once the plan runs out): it adds 10 steps, and
# "crash, saved" saves a checkpoint there first. A crash is SIGKILL, which ends it as a segfault would, without a core.
# The ones that can't stop stand in for a trainer wedged in a GPU call: "deaf" ignores SIGTERM, "save, then deaf" is
# saving a checkpoint for 2 s first, and "dies after Stop" goes 0.3 s after SIGTERM, as a timed-out collective ends it.
TRAINER = """
import json, os, signal, sys, time
from pathlib import Path

run = Path(sys.argv[1])
launches = json.loads((run / "launches.json").read_text()) if (run / "launches.json").exists() else []
launches.append(sys.argv[2:])
(run / "launches.json").write_text(json.dumps(launches))
plan = json.loads((run / "plan.json").read_text())
act = plan[min(len(launches), len(plan)) - 1]


def report(**fields):
    status.update(fields)
    (run / "status.tmp").write_text(json.dumps(status))
    os.replace(run / "status.tmp", run / "status.json")  # the supervisor reads it while this runs


def save():
    folder = run / "checkpoints" / f"step-{status['step']:07d}"
    folder.mkdir(parents=True)
    (folder / "state.json").write_text("{}")


if act in ("deaf", "save, then deaf"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if act == "dies after Stop":
    signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.3), os.kill(os.getpid(), signal.SIGKILL)))
status = json.loads((run / "status.json").read_text())
report(step=status.get("step", 0) + 10, pid=os.getpid(),
       state=act if act in ("done", "failed", "stopped") else "running")
if act == "crash, saved":
    save()
if act.startswith("crash"):
    os.kill(os.getpid(), signal.SIGKILL)
if act == "save, then deaf":
    report(stage="Saving a checkpoint", saving_since=time.time())
    time.sleep(2)
    save()
    (run / "saved").write_text(json.dumps(time.time()))
    report(stage="Training", saving_since=None)  # Stop can end it from here on
if act in ("wait", "deaf", "save, then deaf", "dies after Stop"):
    time.sleep(60)
"""

# Each card of a two-card run under the real torchrun: it hears Stop (and says so) but can't act on it, wedged.
CARD = """
import json, os, signal, sys, time
from pathlib import Path

run, rank = Path(sys.argv[1]), os.environ["RANK"]
signal.signal(signal.SIGTERM, lambda *_: (run / f"heard-{rank}").write_text("SIGTERM"))
if rank == "0":
    status = {**json.loads((run / "status.json").read_text()), "state": "running", "step": 10, "pid": os.getpid()}
    (run / "status.tmp").write_text(json.dumps(status))
    os.replace(run / "status.tmp", run / "status.json")
(run / f"card-{rank}").write_text(str(os.getpid()))
time.sleep(60)
"""


def supervise(run: Path, plan: list[str], command: list[str] | None = None) -> subprocess.Popen:
    run.mkdir(parents=True)
    (run / "trainer.py").write_text(TRAINER)
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "status.json").write_text(json.dumps({"state": "starting", "step": 0}))
    log = (run / "log.txt").open("ab")
    return subprocess.Popen(
        [sys.executable, "-m", "anvil.supervisor", str(run), "--",
         *(command or [sys.executable, str(run / "trainer.py"), str(run)])],
        env={**os.environ, "PYTHONPATH": str(SERVER)}, stdout=log, stderr=subprocess.STDOUT)


def status(run: Path) -> dict:
    return json.loads((run / "status.json").read_text())


def launches(run: Path) -> list[list[str]]:
    return json.loads((run / "launches.json").read_text())


def wait_for(check, timeout=30):
    deadline = time.time() + timeout
    while not check():
        assert time.time() < deadline, "timed out"
        time.sleep(0.05)


def press_stop(proc: subprocess.Popen, run: Path, end_in: float) -> float:
    """What runs.stop does: stop.json says when to end the trainer outright if it hasn't stopped (end_in 0 is Stop
    now), and SIGTERM tells the supervisor. Returns when."""
    now = time.time()
    asked = json.loads((run / "stop.json").read_text()) if (run / "stop.json").exists() else {"requested": now}
    (run / "stop.json").write_text(json.dumps({**asked, "end_by": now + end_in}))
    proc.send_signal(signal.SIGTERM)
    return now


def gone(pid: int) -> bool:
    found = supervisor._stat(pid)
    return found is None or found[1] == "Z"


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
    wait_for(lambda: status(run).get("state") == "running")
    proc.send_signal(signal.SIGTERM)  # a Stop without stop.json, from a shell say
    assert proc.wait(timeout=60) == 0
    st = status(run)
    assert st["state"] == "stopped" and "crashes" not in st and len(launches(run)) == 1
    asked = json.loads((run / "stop.json").read_text())  # written down for the run page, with the usual wait
    assert asked["end_by"] - asked["requested"] == pytest.approx(supervisor.STOP_GRACE_S)


def test_a_trainer_deaf_to_stop_is_ended_when_the_wait_runs_out(tmp_path):
    run = tmp_path / "run"
    proc = supervise(run, ["deaf"])
    wait_for(lambda: status(run).get("state") == "running")
    asked = press_stop(proc, run, end_in=1.5)
    assert proc.wait(timeout=30) == 0
    st = status(run)
    assert st["state"] == "stopped" and "crashes" not in st and len(launches(run)) == 1, st
    assert st["ended"]["at"] >= asked + 1.5 and st["ended"]["step"] == 10 and st["ended"]["from"] is None
    assert gone(st["pid"])
    assert "so it was ended outright (SIGKILL)" in (run / "log.txt").read_text()


def test_stop_now_ends_a_deaf_trainer_at_once(tmp_path):
    run = tmp_path / "run"
    proc = supervise(run, ["deaf"])
    wait_for(lambda: status(run).get("state") == "running")
    press_stop(proc, run, end_in=600)
    with pytest.raises(subprocess.TimeoutExpired):
        proc.wait(timeout=1.5)  # asked to stop after its step, which it never gets to the end of
    now = press_stop(proc, run, end_in=0)
    assert proc.wait(timeout=10) == 0
    st = status(run)
    assert st["state"] == "stopped" and now <= st["ended"]["at"] < now + 2 and len(launches(run)) == 1


def test_a_checkpoint_being_saved_is_never_cut_off(tmp_path):
    run = tmp_path / "run"
    proc = supervise(run, ["save, then deaf"])
    wait_for(lambda: status(run).get("saving_since"))
    press_stop(proc, run, end_in=0)  # Stop now, halfway through the save
    assert proc.wait(timeout=30) == 0
    st = status(run)
    saved = json.loads((run / "saved").read_text())
    assert st["state"] == "stopped" and st["ended"]["at"] >= saved and st["ended"]["from"] == 10, st
    assert (run / "checkpoints" / "step-0000010" / "state.json").exists()


def test_a_trainer_that_dies_after_stop_is_not_started_again(tmp_path):
    run = tmp_path / "run"
    proc = supervise(run, ["dies after Stop", "done"])
    wait_for(lambda: status(run).get("state") == "running")
    press_stop(proc, run, end_in=600)  # it dies long before the wait runs out, without saving
    assert proc.wait(timeout=30) == 0
    st = status(run)
    assert st["state"] == "stopped" and "crashes" not in st and len(launches(run)) == 1, st
    assert st["ended"]["step"] == 10 and "ended after Stop (killed by SIGKILL)" in (run / "log.txt").read_text()


def test_each_card_hears_stop_and_a_deaf_run_is_ended_whole(tmp_path):
    run = tmp_path / "run"
    (tmp_path / "card.py").write_text(CARD)
    proc = supervise(run, [], [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
                               str(tmp_path / "card.py"), str(run)])
    wait_for(lambda: (run / "card-0").exists() and (run / "card-1").exists(), timeout=120)
    cards = [int((run / f"card-{rank}").read_text()) for rank in (0, 1)]
    asked = press_stop(proc, run, end_in=2)
    assert proc.wait(timeout=60) == 0
    # Each card was asked itself: torchrun wasn't, or it would have killed them 30 s later, saving or not.
    assert (run / "heard-0").exists() and (run / "heard-1").exists()
    assert "death signal" not in (run / "log.txt").read_text()
    st = status(run)
    assert st["state"] == "stopped" and st["ended"]["at"] >= asked + 2 and "crashes" not in st, st
    assert all(gone(pid) for pid in cards)
