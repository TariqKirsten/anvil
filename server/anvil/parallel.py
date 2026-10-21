"""Training on several GPUs at once (data parallel).

Each card holds a whole copy of the model and trains on its own share of every batch. After each step the cards
average their gradients, so the copies stay identical. That's how a model that fits on one card uses more of them:
about N times the rows per second, minus a little time spent exchanging gradients.

The server starts one process per card with torchrun (torch.distributed.run), which tells each process its RANK and
the WORLD_SIZE. Process 0 is in charge: it writes the status, the metrics, the leaderboard and the checkpoints, and it
finishes the run (the final test and saving the model) after the others have left. Validation checks are shared: each
card scores its share of the rows and the results are put together.

With one card there's no torchrun, and Group does nothing, so the trainer has a single code path.
"""

from __future__ import annotations

import contextlib
import os
from datetime import timedelta

import torch
import torch.distributed as dist

TIMEOUT = timedelta(minutes=30)  # how long a card waits for the others (process 0 saving a checkpoint, say)


class Group:
    def __init__(self):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.on = self.world > 1
        self.main = self.rank == 0
        self.device = torch.device("cpu")

    def start(self, device: torch.device) -> None:
        self.device = device
        if not self.on or dist.is_initialized():
            return
        if device.type == "cuda":
            torch.cuda.set_device(device)
            dist.init_process_group("nccl", timeout=TIMEOUT, device_id=device)  # RCCL on ROCm
        else:
            dist.init_process_group("gloo", timeout=TIMEOUT)

    def end(self, wait: bool = False) -> None:
        """Leave the group. wait: first wait for every card to get here (the normal way out; not after an error,
        when the others may be stuck elsewhere)."""
        if self.on and dist.is_initialized():
            if wait:
                dist.barrier()
            dist.destroy_process_group()

    def sum(self, values: list[float]) -> list[float]:
        """Each value added up across the cards."""
        if not self.on:
            return [float(v) for v in values]
        t = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        dist.all_reduce(t)
        return t.tolist()

    def any(self, flag: bool) -> bool:
        return self.sum([float(flag)])[0] > 0

    def all(self, flag: bool) -> bool:
        return self.sum([float(not flag)])[0] == 0

    def share(self, value):
        """Process 0's value, on every card."""
        if not self.on:
            return value
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def gather(self, value) -> list:
        """Every card's value, on every card, in rank order."""
        if not self.on:
            return [value]
        out = [None] * self.world
        dist.all_gather_object(out, value)
        return out

    def wrap(self, model, task) -> torch.nn.Module:
        """What the training loop calls for a batch's loss. With several cards it's wrapped in DistributedDataParallel,
        which averages the gradients across the cards during the backward pass."""
        learner = _Loss(model, task)
        if not self.on:
            return learner
        from torch.nn.parallel import DistributedDataParallel

        # Buffers (rotary tables and the like) are the same on every card: no need to resend them every step.
        return DistributedDataParallel(learner, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                                       forward_sync_buffers=False)

    def accumulating(self, learner, sync: bool):
        """On gradient accumulation's in-between batches the cards don't exchange gradients: once per step is enough,
        on the step's last batch."""
        if self.on and not sync:
            return learner.no_sync()
        return contextlib.nullcontext()


class _Loss(torch.nn.Module):
    """The model, the task's own weights (Clef's head) and the task's loss as one module, so DistributedDataParallel
    sees the whole forward pass (the tasks read the model's hidden states directly instead of calling its forward)
    and averages the head's gradients too."""

    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.head = task.head
        self.task = task

    def forward(self, batch: dict) -> torch.Tensor:
        return self.task.loss_sum(self.model, batch)
