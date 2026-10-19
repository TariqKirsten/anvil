"""Keeps a run's trainer going through crashes in native code.

    python -m anvil.supervisor <run_dir> -- <the trainer's command>

The trainer says how a run ended in status.json: done, stopped (Stop) or failed (an error Python caught). A crash in
native code (a segfault, an abort, memory corruption in the GPU libraries) kills it before it can say anything, and so
does the kernel's OOM killer: the status still says it's running. Then this starts it again with --resume, from the
newest checkpoint, so a crash costs the steps since that checkpoint and a minute of loading. With several cards the
trainer is torchrun, which stops every card when one of them dies; the restart comes once they've all gone.

Crashing again and again from the same checkpoint is no fluke: after MAX_TRIES restarts from it the run fails, as it
would have without this, and Resume tries again.

The server starts this in place of the trainer (runs._start). The status's pid is this process until the trainer writes
its own, and again once it has crashed, so Stop reaches it then: it passes Stop on and starts nothing new.
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
STOP = {"requested": False}
_child: subprocess.Popen | None = None


def _ask_to_stop(signum, frame):
    STOP["requested"] = True
    if _child is not None and _child.poll() is None:
        _child.send_signal(signal.SIGTERM)


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


def main() -> None:
    global _child
    run_dir = Path(sys.argv[1])
    command = sys.argv[sys.argv.index("--") + 1:]
    status_path = run_dir / "status.json"
    signal.signal(signal.SIGTERM, _ask_to_stop)
    signal.signal(signal.SIGINT, _ask_to_stop)
    tries, start = 0, None
    while True:
        _child = subprocess.Popen(command)  # the log and the environment (the Hugging Face token) carry over
        if STOP["requested"]:  # Stop came while it was being started
            _child.send_signal(signal.SIGTERM)
        code = _child.wait()
        st = _read(status_path)
        if st.get("state") not in ACTIVE:
            return  # done, stopped or failed: the trainer said how it ended
        if STOP["requested"]:
            st.update(state="stopped", stage="Stopped", eta_s=None, stopped=time.time(), updated=time.time())
            _write(status_path, st)
            return
        newest = newest_checkpoint(run_dir)
        tries = tries + 1 if tries and newest == start else 1  # a newer checkpoint is progress: counting starts again
        start = newest
        where = f"the step-{start:,} checkpoint" if start is not None else "the start"
        crash = {"at": time.time(), "step": st.get("step", 0), "from": start}
        st.update(crashes=st.get("crashes", 0) + 1, last_crash=crash, eta_s=None, updated=time.time())
        if tries > MAX_TRIES:
            st.update(state="failed", stage="Failed",
                      error=f"The trainer crashed {tries} times in a row from {where}, so it was left there. The log "
                            "shows each crash; Resume tries again.")
            _write(status_path, st)
            print(f"\n===== the trainer crashed ({_how(code)}) {tries} times in a row from {where}: giving up "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
            return
        st.update(state="starting", stage="Restarting after a crash", pid=os.getpid())
        _write(status_path, st)
        print(f"\n===== the trainer crashed after step {crash['step']} ({_how(code)}): restarting from {where}, "
              f"try {tries} of {MAX_TRIES}, {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
        if "--resume" not in command:
            command = [*command, "--resume"]


if __name__ == "__main__":
    main()
