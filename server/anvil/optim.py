"""AdamW for weights kept in bf16: the maths in fp32, the results rounded stochastically.

Full fine-tuning keeps the model's weights in bf16 so a 2B model fits a 32 GB card (about 8 bytes a parameter with
AdamW's two moments, against 16 in fp32). But bf16 has 8 bits of mantissa: a step of 2e-5 on a weight near 0.02 is
below its resolution there, so a plain optimizer rounds most small updates away (on Qwen3.5-0.8B half the weights
didn't move at all in a step at 2e-5) and rounds the rest up to a whole unit. The same happens to Adam's second
moment, whose update each step is a thousandth of its size.

Stochastic rounding is the usual fix: a value between two bf16 numbers goes to either with probability given by where
it lies, so the rounded weight is right on average and small updates add up over steps instead of vanishing. The
weights, the moments and the update are worked out in fp32 (in slices, so an embedding table doesn't need gigabytes
of scratch) and written back to bf16 with that rounding. On several cards every copy rounds the same way: the
generator is seeded alike on each, so the copies stay identical. fp32 parameters (LoRA's adapters, Clef's head) get
the same maths with no rounding to do, and match torch's AdamW.
"""

from __future__ import annotations

import math

import torch

SLICE = 1 << 24  # parameters worked on at once: 64 MB of fp32 scratch per temporary


def round_to_bf16_(out: torch.Tensor, value: torch.Tensor, generator: torch.Generator) -> None:
    """Write fp32 `value` into bf16 `out`, rounding each number up or down at random so it's right on average. bf16
    is fp32's top 16 bits: adding a random 16-bit number to the bits below, then cutting them off, rounds up with
    the probability the cut-off part would have had."""
    bits = value.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32, generator=generator)
    out.copy_(((bits + noise) & -65536).view(torch.float32))  # & -65536 clears the low 16 bits; the cast is exact


class AdamW(torch.optim.Optimizer):
    """torch.optim.AdamW's update (decoupled weight decay, bias correction), computed in fp32 for parameters of any
    floating type, with stochastic rounding into bf16 parameters and moments."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01,
                 generator: torch.Generator | None = None):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})
        self.generator = generator

    # The rounding generator's state goes into checkpoints with the moments, so a run resumed from one rounds on
    # exactly as it would have without stopping.
    def state_dict(self):
        found = super().state_dict()
        if self.generator is not None:
            found["rounding"] = self.generator.get_state()
        return found

    def load_state_dict(self, state_dict):
        rounding = state_dict.get("rounding")
        super().load_state_dict({k: v for k, v in state_dict.items() if k != "rounding"})
        if rounding is not None and self.generator is not None:
            self.generator.set_state(rounding.cpu())  # a checkpoint loaded onto the GPU brings it there: it lives on the CPU

    def _generator(self, device: torch.device) -> torch.Generator:
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(0)
        return self.generator

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), eps, decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                scale1, scale2 = 1 - beta1 ** t, math.sqrt(1 - beta2 ** t)  # bias corrections
                stochastic = p.dtype == torch.bfloat16
                flat = [x.view(-1) for x in (p, p.grad, state["exp_avg"], state["exp_avg_sq"])]
                for start in range(0, p.numel(), SLICE):
                    w, g, m, v = (x[start:start + SLICE] for x in flat)
                    g32 = g.float()
                    m32 = m.float().mul_(beta1).add_(g32, alpha=1 - beta1)
                    v32 = v.float().mul_(beta2).addcmul_(g32, g32, value=1 - beta2)
                    w32 = w.float()
                    if decay:
                        w32.mul_(1 - lr * decay)
                    w32.addcdiv_(m32, (v32.sqrt() / scale2).add_(eps), value=-lr / scale1)
                    if stochastic:
                        generator = self._generator(p.device)
                        for dst, src in ((w, w32), (m, m32), (v, v32)):
                            round_to_bf16_(dst, src, generator)
                    else:
                        w.copy_(w32)
                        m.copy_(m32)
                        v.copy_(v32)
        return loss
