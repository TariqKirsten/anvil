"""Keeps a run's trainer going through crashes in native code, and sees Stop through even when the trainer can't.

    python -m anvil.supervisor <run_dir> -- <the trainer's command>

The trainer says how a run ended in status.json: done, stopped (Stop) or failed (an error Python caught). A crash in
native code (a segfault, an abort, memory corruption in the GPU libraries) kills it before it can say anything, and so
does the kernel's OOM killer: the status still says it's running. Then this starts it again with --resume, from the
newest checkpoint, so a crash costs the steps since that checkpoint and a minute of loading. With several cards the
trainer is torchrun, which stops every card when one of them dies; the restart comes once they've all gone.

Crashing again and again from the same checkpoint is no fluke: after MAX_TRIES restarts from it the run fails, as it
would have without this, and Resume tries again.

A trainer can also freeze without crashing: a card stops answering (its firmware hangs, say), the GPU call waiting on
it never returns, and the cards go idle while the status still says it's running. Nothing errors, so nothing else
notices. The trainer rewrites its status at least every minute while it works (anvil.train's BEAT_S), so one that has
said nothing for STALL_S has frozen: it's ended outright, every process of it, which is also what gets the kernel to
reset a hung card, and started again like after a crash, FREEZE_PAUSE_S later. A checkpoint being saved gets
SAVE_LIMIT_S, and loading the model isn't watched (a download takes as long as it takes).

A run can switch the restarts off, from the form or the run page at any time (config.json's supervise, read when it
matters): then a crash fails the run, as it would without this, and a freeze is left for Stop, which works either way.

Stop comes here as stop.json, which runs.stop writes (a SIGTERM to this process, from a shell say, is a Stop too). This
asks the trainer to stop after its current step, with a SIGTERM to it, or with several cards to each card's process: not
to torchrun, which would pass it on and kill them 30 s later, perhaps halfway through saving the checkpoint. The trainer
saves a checkpoint and says it stopped. One that's wedged, in a GPU collective say, never gets to the end of its step,
so at stop.json's end_by (STOP_GRACE_S after Stop, or at once for Stop now) it's ended outright, every process of it,
and the run is stopped all the same: Resume continues from the newest checkpoint. A checkpoint being saved is never cut
off (status.json's saving_since), unless it has taken SAVE_LIMIT_S, which no save takes. And once Stop has come nothing
is started again, however the trainer ends.

The server starts this in place of the trainer (runs._start). The status's pid is this process until the trainer writes
its own, and again once it has crashed.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ACTIVE = {"starting", "loading", "running", "saving"}  # as in runs.ACTIVE: the trainer hasn't said how it ended
MAX_TRIES = 3  # restarts from one checkpoint before giving up
STOP_GRACE_S = 10 * 60  # after Stop, time to finish the step (seconds, minutes with a check) before it's ended outright
SAVE_LIMIT_S = 30 * 60  # a save still going after this long is stuck (the other cards give up waiting for one then too)
KILL_WAIT_S = 30  # how long ended processes get to go (one stuck in a GPU driver call may never)
STALL_S = 15 * 60  # a working trainer that hasn't touched its status for this long has frozen
FREEZE_PAUSE_S = 15  # after ending a frozen trainer, time for the kernel to reset a hung card before starting again
SIGNALLED: dict = {}  # a SIGTERM's Stop: when, and when to end the trainer (stop.json's, if there's one, comes first)
_child: subprocess.Popen | None = None


def _ask_to_stop(signum, frame):
    if not SIGNALLED:  # carried out by _watch within half a second
        now = time.time()
        SIGNALLED.update(requested=now, end_by=now + STOP_GRACE_S)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def newest_checkpoint(run_dir: Path) -> int | None:
    """The step of the newest whole checkpoint, which is where --resume starts (None: there's none yet)."""
    steps = [int(p.name[5:]) for p in (run_dir / "checkpoints").glob("step-*")
             if p.name[5:].isdigit() and (p / "state.json").exists()]  # not step-N.tmp, still being written
    return max(steps, default=None)


def _how(code: int) -> str:
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit code {code}"


def _when(seconds: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(seconds))


# ---------------------------------------------------------------- Stop

def _stop(run_dir: Path) -> dict | None:
    """The Stop under way: when it was asked for, and when to end the trainer outright if it hasn't stopped by itself
    (None: there's none)."""
    stop = _read(run_dir / "stop.json")
    if not stop and SIGNALLED:  # a SIGTERM without stop.json, from a shell say: written down for the run page
        stop = dict(SIGNALLED)
        _write(run_dir / "stop.json", stop)
    return stop or None


def _saving(run_dir: Path) -> bool:
    """Whether the trainer is saving a checkpoint (or the finished model): Stop waits for that, unless it's stuck."""
    since = _read(run_dir / "status.json").get("saving_since")
    return bool(since) and time.time() - since < SAVE_LIMIT_S


def _on(run_dir: Path) -> bool:
    """Whether this is on for the run: config.json's supervise, which the run page can switch at any time (runs from
    before it have none, and it's on). Off, a crash fails the run and a freeze waits for Stop."""
    return _read(run_dir / "config.json").get("supervise", True) is not False


def _frozen(run_dir: Path) -> dict | None:
    """The trainer's status if it has frozen: working, not saving, and silent for STALL_S (None: it hasn't)."""
    st = _read(run_dir / "status.json")
    if st.get("state") not in ("running", "saving") or _saving(run_dir):
        return None
    return st if time.time() - st.get("updated", time.time()) > STALL_S else None


def _stat(pid: int) -> tuple[int, str, int] | None:
    """A process's parent, state (Z: it has ended) and start time, from /proc (None: it's gone)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()  # after the name, which may hold spaces and brackets
    except (OSError, IndexError):
        return None
    return int(fields[1]), fields[0], int(fields[19])


def _processes() -> dict[int, tuple[int, str, int]]:
    return {int(name): info for name in os.listdir("/proc") if name.isdigit() and (info := _stat(int(name)))}


def _under(root: int) -> dict[int, int]:
    """root and every process under it, with their start times: with several cards torchrun and each card's process,
    which torchrun starts in a session of its own (so they aren't in this process group)."""
    procs = _processes()
    children: dict[int, list[int]] = {}
    for pid, (parent, _, _) in procs.items():
        children.setdefault(parent, []).append(pid)
    found, todo = {}, [root]
    while todo:
        pid = todo.pop(0)
        if pid in procs and pid not in found:
            found[pid] = procs[pid][2]
            todo += children.get(pid, [])
    return found


def _send(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _ask(several: bool) -> None:
    """Ask the trainer to stop after this step: SIGTERM to it, or with several cards to each card's process."""
    cards = [pid for pid, (parent, _, _) in _processes().items() if parent == _child.pid] if several else []
    for pid in cards or [_child.pid]:  # no cards yet: torchrun is still starting, and nothing has trained
        _send(pid, signal.SIGTERM)


def _end() -> list[int]:
    """End the trainer outright: SIGKILL to it and every process under it, and wait for them to go. Returns those still
    there after KILL_WAIT_S, stuck in the kernel (a hung GPU driver call): nothing ends those, and the log says so."""
    procs = _under(_child.pid)
    for pid in procs:  # torchrun first, then the cards
        _send(pid, signal.SIGKILL)
    deadline = time.time() + KILL_WAIT_S
    while True:
        _child.poll()  # reaps it; the cards' processes, orphaned, are reaped by init
        left = [pid for pid, started in procs.items()
                if (info := _stat(pid)) and info[1] != "Z" and info[2] == started]  # the same process, still there
        if not left or time.time() > deadline:
            for pid in left:
                print(f"===== process {pid} is still there after SIGKILL: stuck in the GPU driver? Its card may need "
                      "a reset or a reboot before it trains again =====", flush=True)
            return left
        time.sleep(0.2)


def _watch(run_dir: Path, several: bool) -> tuple[int | None, float | None, dict | None]:
    """Wait for the trainer to end, carry out Stop if it comes, and end the trainer if it freezes. Returns its exit code
    (None: it wouldn't go even when killed), when Stop had to end it outright, and if it froze, where and whether it
    could be ended."""
    asked = False
    while True:
        try:
            return _child.wait(timeout=0.5), None, None
        except subprocess.TimeoutExpired:
            pass
        stop = _stop(run_dir)
        if stop is None:
            st = _frozen(run_dir)
            if st and _on(run_dir):
                print(f"\n===== the trainer froze: nothing from it since {_when(st['updated'])}, at step "
                      f"{st.get('step', 0)}, so it was ended outright (SIGKILL), {_when(time.time())} =====",
                      flush=True)
                left = _end()
                return _child.poll(), None, {"step": st.get("step", 0), "stuck": bool(left)}
            continue
        if not asked:
            _ask(several)
            asked = True
            print(f"\n===== Stop: the trainer was asked to stop after this step, {_when(time.time())} =====",
                  flush=True)
        if time.time() >= stop.get("end_by", 0) and not _saving(run_dir):
            ended = time.time()
            print(f"\n===== Stop: the trainer hadn't stopped by itself, so it was ended outright (SIGKILL), "
                  f"{_when(ended)} =====", flush=True)
            _end()
            return _child.poll(), ended, None


def main() -> None:
    global _child
    run_dir = Path(sys.argv[1])
    command = sys.argv[sys.argv.index("--") + 1:]
    several = "torch.distributed.run" in command  # torchrun, which starts a process for each card
    status_path = run_dir / "status.json"
    signal.signal(signal.SIGTERM, _ask_to_stop)
    signal.signal(signal.SIGINT, _ask_to_stop)
    (run_dir / "stop.json").unlink(missing_ok=True)  # an old Stop's (runs._spawn removes them too); a new one signals
    tries, start, how = 0, None, set()
    while True:
        _child = subprocess.Popen(command)  # the log and the environment (the Hugging Face token) carry over
        code, ended, froze = _watch(run_dir, several)
        st = _read(status_path)
        if st.get("state") not in ACTIVE:
            return  # done, stopped or failed: the trainer said how it ended
        newest = newest_checkpoint(run_dir)
        where = f"the step-{newest:,} checkpoint" if newest is not None else "the start"
        if _stop(run_dir) is not None:  # however the trainer ended, after Stop it stays stopped
            now = time.time()
            st.update(state="stopped", stage="Stopped", eta_s=None, saving_since=None, stopped=now, updated=now,
                      ended={"at": ended or now, "step": st.get("step", 0), "from": newest})
            _write(status_path, st)
            if not ended:
                print(f"\n===== the trainer ended after Stop ({_how(code)}) without saving, {_when(now)} =====",
                      flush=True)
            print(f"===== stopped: Resume continues from {where} =====", flush=True)
            return
        if not froze and not _on(run_dir):  # switched off: the crash ends the run, as it would without this
            st.update(state="failed", stage="Failed", eta_s=None, updated=time.time(),
                      error=f"The trainer crashed after step {st.get('step', 0):,} ({_how(code)}). The supervisor is "
                            f"off for this run, so it wasn't started again: Resume continues from {where}.")
            _write(status_path, st)
            print(f"\n===== the trainer crashed after step {st.get('step', 0)} ({_how(code)}), and the supervisor is "
                  f"off: not starting it again, {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
            return
        tries = tries + 1 if tries and newest == start else 1  # a newer checkpoint is progress: counting starts again
        how = (how if tries > 1 else set()) | {"froze" if froze else "crashed"}
        start = newest
        crash = {"at": time.time(), "step": st.get("step", 0), "from": start, **({"froze": True} if froze else {})}
        st.update(crashes=st.get("crashes", 0) + 1, last_crash=crash, eta_s=None, updated=time.time())
        if froze and froze["stuck"]:  # its card wouldn't take a new trainer either
            st.update(state="failed", stage="Failed",
                      error=f"The trainer froze after step {crash['step']:,} and couldn't be ended: it's stuck in the "
                            f"GPU driver, so its card needs a reset or a reboot. Then Resume continues from {where}.")
            _write(status_path, st)
            print(f"\n===== the frozen trainer couldn't be ended: giving up {time.strftime('%Y-%m-%d %H:%M:%S')} =====",
                  flush=True)
            return
        if tries > MAX_TRIES:
            what = " or ".join(sorted(how))  # crashed, froze, or crashed or froze
            st.update(state="failed", stage="Failed",
                      error=f"The trainer {what} {tries} times in a row from {where}, so it was left there. The log "
                            "shows each time; Resume tries again.")
            _write(status_path, st)
            detail = "" if froze else f" ({_how(code)})"
            print(f"\n===== the trainer {what}{detail} {tries} times in a row from {where}: giving up "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
            return
        st.update(state="starting", stage=f"Restarting after a {'freeze' if froze else 'crash'}", pid=os.getpid())
        _write(status_path, st)
        if froze:
            print(f"\n===== the trainer froze after step {crash['step']}: restarting from {where} in "
                  f"{FREEZE_PAUSE_S} s, try {tries} of {MAX_TRIES}, {time.strftime('%Y-%m-%d %H:%M:%S')} =====",
                  flush=True)
            time.sleep(FREEZE_PAUSE_S)
        else:
            print(f"\n===== the trainer crashed after step {crash['step']} ({_how(code)}): restarting from {where}, "
                  f"try {tries} of {MAX_TRIES}, {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
        if "--resume" not in command:
            command = [*command, "--resume"]


if __name__ == "__main__":
    main()
