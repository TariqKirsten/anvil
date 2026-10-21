"""AdamW with stochastic rounding (anvil.optim): right on average in bf16, and torch's AdamW in fp32."""

import torch

from anvil import optim


def test_stochastic_rounding_is_right_on_average_and_lands_on_a_neighbour():
    generator = torch.Generator().manual_seed(0)
    value = torch.full((20000,), 1.0 + 0.3 * 2 ** -7)  # 30% of the way from 1 to the next bf16 number, 1 + 2^-7
    out = torch.empty_like(value, dtype=torch.bfloat16)
    optim.round_to_bf16_(out, value, generator)
    assert set(out.unique().tolist()) == {1.0, 1.0 + 2 ** -7}
    assert abs(out.float().mean().item() - value[0].item()) < 1e-4  # about 30% round up
    negative = torch.full((20000,), -(1.0 + 0.3 * 2 ** -7))
    optim.round_to_bf16_(out, negative, generator)
    assert set(out.unique().tolist()) == {-1.0, -(1.0 + 2 ** -7)} and abs(out.float().mean().item() - negative[0].item()) < 1e-4
    exact = torch.tensor([0.5, -3.0, 0.0, 1024.0])
    optim.round_to_bf16_(out[:4], exact, generator)
    assert out[:4].float().tolist() == exact.tolist()  # a number bf16 holds exactly never moves


def test_fp32_parameters_follow_torchs_adamw():
    torch.manual_seed(0)
    ours, theirs = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
    theirs.load_state_dict(ours.state_dict())
    x = torch.randn(16, 8)
    a = optim.AdamW(ours.parameters(), lr=1e-2, weight_decay=0.1)
    b = torch.optim.AdamW(theirs.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(20):
        for model, opt in ((ours, a), (theirs, b)):
            opt.zero_grad()
            (model(x) ** 2).mean().backward()
            opt.step()
    assert torch.allclose(ours.weight, theirs.weight, atol=1e-6) and torch.allclose(ours.bias, theirs.bias, atol=1e-6)


def test_small_updates_add_up_in_bf16_instead_of_rounding_away():
    # A weight near 1 has a bf16 resolution of 2^-7: a step of 1e-3 is a ninth of that. Over 200 steps the weight
    # should move by about 0.2; a plain optimizer leaves it where it was.
    weight = torch.nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
    opt = optim.AdamW([weight], lr=1e-3, weight_decay=0.0, generator=torch.Generator().manual_seed(1))
    plain = torch.nn.Parameter(torch.ones(4096, dtype=torch.bfloat16))
    torch_opt = torch.optim.AdamW([plain], lr=1e-3, weight_decay=0.0)
    for _ in range(200):
        for p, o in ((weight, opt), (plain, torch_opt)):
            o.zero_grad()
            p.grad = torch.ones_like(p)  # Adam's step is then lr, downhill
            o.step()
    moved = 1 - weight.float().mean().item()
    assert 0.18 < moved < 0.22, moved
    assert plain.float().mean().item() == 1.0  # torch's AdamW on bf16: every update rounded away
    assert opt.state[weight]["exp_avg"].dtype == torch.bfloat16 and opt.state[weight]["step"] == 200


def test_the_optimizer_resumes_from_its_state_dict():
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(1000, dtype=torch.bfloat16))
    opt = optim.AdamW([weight], lr=1e-2, generator=torch.Generator().manual_seed(2))
    for _ in range(5):
        weight.grad = torch.randn_like(weight)
        opt.step()
    saved = {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state[weight].items()}
    again = optim.AdamW([weight], lr=1e-2, generator=torch.Generator().manual_seed(2))
    again.load_state_dict(opt.state_dict())
    state = again.state[weight]
    assert state["step"] == 5 and torch.equal(state["exp_avg"], saved["exp_avg"]) and state["exp_avg"].dtype == torch.bfloat16
    weight.grad = torch.randn_like(weight)
    again.step()
    assert again.state[weight]["step"] == 6
