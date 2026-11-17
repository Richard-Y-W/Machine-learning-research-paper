"""Dynamics, exact network symmetries, and transparent numerical edit baselines."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, jacrev

Tensor = torch.Tensor


def features(x: Tensor, family: str) -> Tensor:
    q, v, u = x.unbind(-1)
    if family == "pendulum":
        return torch.stack((-q.sin(), -v, -v * v.abs(), u), -1)
    return torch.stack((-q, -v, -q**3, u), -1)


def coefficients(seed: int, family: str) -> Tensor:
    rng = np.random.default_rng(50000 + seed)
    c = np.array([rng.uniform(.7, 1.3), rng.uniform(.2, .6),
                  rng.uniform(.1, .4), rng.uniform(.8, 1.2)])
    return torch.tensor(c, dtype=torch.float64)


def acceleration(x: Tensor, c: Tensor, family: str) -> Tensor:
    y = features(x, family) @ c.to(x)
    if family == "duffing_mismatch":
        y = y + .3 * torch.sin(3 * x[..., 0])
    return y


def intervention(name: str) -> Tensor:
    return torch.tensor({"damping": [0., 1., 0., 0.],
                         "restoring": [.5, 0., 0., 0.],
                         "combined": [.5, 1., 0., 0.]}[name], dtype=torch.float64)


class DynamicsNet(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

    def hidden(self, x: Tensor) -> Tensor:
        return self.net[:4](x)


def sample(n: int, seed: int, extent: float = 1.5) -> Tensor:
    return torch.tensor(np.random.default_rng(seed).uniform(-extent, extent, (n, 3)),
                        dtype=torch.float64)


def train(seed: int, family: str, width: int, steps: int) -> tuple[DynamicsNet, dict]:
    torch.manual_seed(1000 + seed)
    model = DynamicsNet(width).float()
    x = sample(4096, 10000 + seed).float()
    y = acceleration(x, coefficients(seed, family), family)
    gen = torch.Generator().manual_seed(20000 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for step in range(steps):
        idx = torch.randint(len(x), (256,), generator=gen)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x[idx]) - y[idx]).square().mean()
        loss.backward()
        optimizer.step()
        if step == int(steps * .75):
            for group in optimizer.param_groups:
                group["lr"] *= .2
    model = model.double().eval()
    test = sample(2048, 30000 + seed, 1.25)
    with torch.no_grad():
        rmse = (model(test) - acceleration(test, coefficients(seed, family), family)).square().mean().sqrt()
    return model, {"train_loss_last_batch": float(loss.detach()), "base_rmse": float(rmse)}


def reparameterize(model: DynamicsNet, seed: int, kind: str) -> DynamicsNet:
    out = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for i in (0, 2):
            layer, nxt = out.net[i], out.net[i + 2]
            n = layer.out_features
            if kind == "permuted":
                p = torch.randperm(n, generator=gen)
                layer.weight.copy_(layer.weight[p].clone())
                layer.bias.copy_(layer.bias[p].clone())
                nxt.weight.copy_(nxt.weight[:, p].clone())
            elif kind == "rescaled":
                scale = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(
                    -np.log(10), np.log(10), generator=gen))
                layer.weight.mul_(scale[:, None])
                layer.bias.mul_(scale)
                nxt.weight.div_(scale[None, :])
            elif kind != "original":
                raise ValueError(kind)
    return out


def fit_coefficients(x: Tensor, y: Tensor, family: str) -> Tensor:
    return torch.linalg.lstsq(features(x, family), y, driver="gelsd").solution


def jacobian(model: DynamicsNet, x: Tensor) -> Tensor:
    params = dict(model.named_parameters())
    def forward(p):
        return functional_call(model, p, (x,))
    blocks = jacrev(forward)(params)
    return torch.cat([blocks[k].reshape(len(x), -1) for k in params], dim=1).detach()


def parameter_vector(model: DynamicsNet) -> Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_vector(model: DynamicsNet, vector: Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(vector[offset:offset+p.numel()].reshape_as(p))
            offset += p.numel()
    if offset != len(vector):
        raise ValueError("Wrong parameter vector length")


def ridge_step(j: Tensor, residual: Tensor, normalized: bool, ridge: float) -> Tensor:
    if normalized:
        norms = j.square().sum(0).sqrt()
        # Exact zero columns are inactive; relative threshold avoids division overflow.
        active = norms > torch.finfo(j.dtype).tiny**.25
        scale = torch.where(active, norms, torch.ones_like(norms))
    else:
        scale = torch.ones(j.shape[1], dtype=j.dtype)
    a = j / scale
    kernel = a @ a.T
    lam = ridge * kernel.diag().mean().clamp_min(1e-15)
    alpha = torch.linalg.solve(kernel + lam * torch.eye(len(j), dtype=j.dtype), residual)
    return (a.T @ alpha) / scale


@dataclass
class EditResult:
    predict: Callable[[Tensor], Tensor]
    jacobian_evaluations: int = 0
    optimizer_forward_points: int = 0
    parameter_count: int = 0
    changed_parameters: int = 0
    model: DynamicsNet | None = None


def weight_edit(model: DynamicsNet, x: Tensor, target: Tensor, method: str,
                ridge: float = 1e-4, iterations: int = 3) -> EditResult:
    out = copy.deepcopy(model)
    original = parameter_vector(out)
    nj, nf = 0, 0
    if method == "finetune_adam":
        optimizer = torch.optim.Adam(out.parameters(), lr=.003)
        for _ in range(150):
            optimizer.zero_grad(set_to_none=True)
            loss = (out(x)-target).square().mean()
            loss.backward()
            optimizer.step()
            nf += len(x)
    elif method == "head_ridge":
        with torch.no_grad():
            h = out.hidden(x)
            h = torch.cat((h, torch.ones(len(h), 1, dtype=h.dtype)), 1)
            delta = ridge_step(h, target - out(x), False, ridge)
            out.net[4].weight.add_(delta[:-1][None])
            out.net[4].bias.add_(delta[-1:])
        nf += 2 * len(x)  # separate hidden extraction and prediction calls above
    else:
        for _ in range(iterations):
            with torch.no_grad():
                residual = target - out(x)
                old_loss = float(residual.square().mean())
            nf += len(x)
            j = jacobian(out, x)
            nj += len(x)
            delta = ridge_step(j, residual, method == "normalized_jacobian", ridge)
            before = parameter_vector(out)
            accepted = False
            for alpha in (1., .5, .25, .125, .0625):
                set_vector(out, before + alpha * delta)
                with torch.no_grad():
                    loss = float((out(x) - target).square().mean())
                nf += len(x)
                if loss < old_loss:
                    accepted = True
                    break
            if not accepted:
                set_vector(out, before)
                break
    changed = int(((parameter_vector(out) - original).abs() > 1e-12).sum())
    return EditResult(out, nj, nf, len(original), changed, out)


def rollout(predict: Callable[[Tensor], Tensor], initial: Tensor,
            actions: Tensor, dt: float = .03) -> Tensor:
    state = initial.clone()
    states = []
    with torch.no_grad():
        for u in actions:
            x = torch.cat((state, u[:, None]), -1)
            v = state[:, 1] + dt * predict(x)
            q = state[:, 0] + dt * v
            state = torch.stack((q, v), -1)
            states.append(state)
    return torch.stack(states)
