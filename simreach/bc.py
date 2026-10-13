"""Behavior cloning: a small MLP from privileged state to joint velocity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from simreach.constants import ACTION_DIM, OBS_DIM
from simreach.dataset import Episode, stack_bc_batch
from simreach.normalize import Standardizer


class BCPolicy(nn.Module):
    def __init__(self, hidden: int = 128, obs_dim: int = OBS_DIM, act_dim: int = ACTION_DIM) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class BCCheckpoint:
    model: BCPolicy
    standardizer: Standardizer
    losses: list[float]


def train_bc(
    episodes: list[Episode],
    epochs: int,
    seed: int = 0,
    hidden: int = 128,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    obs_noise: float = 0.02,
) -> BCCheckpoint:
    observations, actions = stack_bc_batch(episodes)
    standardizer = Standardizer.fit(observations)
    inputs = standardizer.transform(observations)
    targets = actions.astype(np.float32)
    torch.manual_seed(seed)
    model = BCPolicy(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    losses = [
        _epoch(model, inputs, targets, optimizer, batch_size, rng, obs_noise=obs_noise)
        for _ in range(epochs)
    ]
    return BCCheckpoint(model=model, standardizer=standardizer, losses=losses)


def _epoch(
    model: BCPolicy,
    inputs: np.ndarray,
    targets: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: np.random.Generator,
    obs_noise: float = 0.0,
) -> float:
    order = rng.permutation(len(inputs))
    total = 0.0
    seen = 0
    model.train()
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        batch_inputs = inputs[batch]
        if obs_noise > 0:
            batch_inputs = batch_inputs + rng.normal(0.0, obs_noise, size=batch_inputs.shape).astype(np.float32)
        pred = model(torch.from_numpy(np.ascontiguousarray(batch_inputs)))
        loss = nn.functional.mse_loss(pred, torch.from_numpy(targets[batch]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(batch)
        seen += len(batch)
    return total / max(seen, 1)


def save_bc(checkpoint: BCCheckpoint, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "bc",
            "state_dict": checkpoint.model.state_dict(),
            "hidden": checkpoint.model.hidden,
            "standardizer": checkpoint.standardizer.to_dict(),
            "losses": checkpoint.losses,
        },
        out,
    )


def load_bc(path: str | Path) -> BCCheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = BCPolicy(hidden=int(payload["hidden"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return BCCheckpoint(
        model=model,
        standardizer=Standardizer.from_dict(payload["standardizer"]),
        losses=[float(value) for value in payload["losses"]],
    )


def bc_action(checkpoint: BCCheckpoint, obs: np.ndarray) -> np.ndarray:
    checkpoint.model.eval()
    normalized = checkpoint.standardizer.transform(np.asarray(obs, dtype=np.float32).reshape(1, -1))
    with torch.no_grad():
        action = checkpoint.model(torch.from_numpy(normalized))
    return action.squeeze(0).numpy().astype(np.float32)
