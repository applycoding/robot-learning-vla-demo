"""Vision-language-action policies.

The default encoder is a small convolutional network plus a learned instruction
embedding, trained on the demonstrations. ``ClipActionModel`` keeps the same
action head but consumes frozen pretrained vision and text embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from simreach.constants import ACTION_DIM, PROPRIO_DIM
from simreach.dataset import Episode, stack_vla_batch
from simreach.normalize import Standardizer


class ActionHead(nn.Module):
    """Map visual features, text features, and proprioception to a joint command."""

    def __init__(self, visual_dim: int, text_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.hidden = int(hidden)
        self.proprio = nn.Sequential(
            nn.Linear(PROPRIO_DIM, 64),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + text_dim + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(
        self, visual: torch.Tensor, text: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([visual, text, self.proprio(proprio)], dim=-1)
        return self.mlp(features)


class TinyVisionEncoder(nn.Module):
    """Task-trained encoder that keeps a coarse spatial grid."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(32 * 4 * 4, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(images).flatten(1))


class TinyVLA(nn.Module):
    """Image + instruction id + proprioception -> normalized joint velocity."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.vision = TinyVisionEncoder(out_dim=128)
        self.text = nn.Embedding(2, 64)
        self.head = ActionHead(self.vision.out_dim, 64, hidden=hidden)

    def forward(
        self, images: torch.Tensor, instruction_ids: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        visual = self.vision(images)
        text = self.text(instruction_ids)
        return self.head(visual, text, proprio)


class ClipActionModel(nn.Module):
    """Trainable action head on frozen vision/text embeddings (CLIP-sized features)."""

    def __init__(self, embed_dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden = int(hidden)
        self.head = ActionHead(embed_dim, embed_dim, hidden=hidden)

    def forward(
        self, visual: torch.Tensor, text: torch.Tensor, proprio: torch.Tensor
    ) -> torch.Tensor:
        return self.head(visual, text, proprio)


@dataclass
class VLACheckpoint:
    kind: str
    model: nn.Module
    proprio_standardizer: Standardizer
    losses: list[float]


def images_to_tensor(images: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(images).copy()
    tensor = torch.from_numpy(array)
    return tensor.permute(0, 3, 1, 2).float().div(255.0)


def train_tiny_vla(
    episodes: list[Episode],
    epochs: int,
    seed: int = 0,
    hidden: int = 128,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> VLACheckpoint:
    images, instruction_ids, proprio, actions = stack_vla_batch(episodes)
    standardizer = Standardizer.fit(proprio)
    proprio_n = standardizer.transform(proprio)
    torch.manual_seed(seed)
    model = TinyVLA(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = [
        _epoch_tiny(
            model,
            images,
            instruction_ids,
            proprio_n,
            actions,
            optimizer,
            batch_size,
            rng,
        )
        for _ in range(epochs)
    ]
    return VLACheckpoint(
        kind="tiny",
        model=model,
        proprio_standardizer=standardizer,
        losses=losses,
    )


def train_embedding_head(
    visual: np.ndarray,
    text: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    seed: int = 0,
    lr: float = 1e-3,
    batch_size: int = 64,
    hidden: int = 128,
) -> VLACheckpoint:
    """Fit an action head on precomputed vision and text embeddings."""
    standardizer = Standardizer.fit(proprio)
    proprio_n = standardizer.transform(proprio)
    torch.manual_seed(seed)
    model = ClipActionModel(embed_dim=int(visual.shape[1]), hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = [
        _epoch_embeddings(
            model,
            visual.astype(np.float32),
            text.astype(np.float32),
            proprio_n,
            actions.astype(np.float32),
            optimizer,
            batch_size,
            rng,
        )
        for _ in range(epochs)
    ]
    return VLACheckpoint(
        kind="clip",
        model=model,
        proprio_standardizer=standardizer,
        losses=losses,
    )


def _epoch_tiny(
    model: TinyVLA,
    images: np.ndarray,
    instruction_ids: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    order = rng.permutation(len(actions))
    total = 0.0
    seen = 0
    model.train()
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        pred = model(
            images_to_tensor(images[batch]),
            torch.from_numpy(instruction_ids[batch]),
            torch.from_numpy(proprio[batch]),
        )
        loss = nn.functional.mse_loss(pred, torch.from_numpy(actions[batch]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(batch)
        seen += len(batch)
    return total / max(seen, 1)


def _epoch_embeddings(
    model: ClipActionModel,
    visual: np.ndarray,
    text: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    order = rng.permutation(len(actions))
    total = 0.0
    seen = 0
    model.train()
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        pred = model(
            torch.from_numpy(visual[batch]),
            torch.from_numpy(text[batch]),
            torch.from_numpy(proprio[batch]),
        )
        loss = nn.functional.mse_loss(pred, torch.from_numpy(actions[batch]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(batch)
        seen += len(batch)
    return total / max(seen, 1)


def save_vla(checkpoint: VLACheckpoint, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = checkpoint.model
    if checkpoint.kind == "tiny":
        assert isinstance(model, TinyVLA)
        architecture: dict[str, int] = {"hidden": model.hidden}
    elif checkpoint.kind == "clip":
        assert isinstance(model, ClipActionModel)
        architecture = {"hidden": model.hidden, "embed_dim": model.embed_dim}
    else:
        raise ValueError(f"unknown VLA kind {checkpoint.kind}")
    torch.save(
        {
            "kind": checkpoint.kind,
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "standardizer": checkpoint.proprio_standardizer.to_dict(),
            "losses": checkpoint.losses,
        },
        out,
    )


def load_vla(path: str | Path) -> VLACheckpoint:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    kind = str(payload["kind"])
    architecture = payload["architecture"]
    if kind == "tiny":
        model: nn.Module = TinyVLA(hidden=int(architecture["hidden"]))
    elif kind == "clip":
        model = ClipActionModel(
            embed_dim=int(architecture["embed_dim"]),
            hidden=int(architecture["hidden"]),
        )
    else:
        raise ValueError(f"unknown VLA kind {kind}")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return VLACheckpoint(
        kind=kind,
        model=model,
        proprio_standardizer=Standardizer.from_dict(payload["standardizer"]),
        losses=[float(value) for value in payload["losses"]],
    )


def tiny_vla_action(checkpoint: VLACheckpoint, image: np.ndarray, instruction_id: int, proprio: np.ndarray) -> np.ndarray:
    if not isinstance(checkpoint.model, TinyVLA):
        raise TypeError("tiny_vla_action expects a tiny VLA checkpoint")
    checkpoint.model.eval()
    normalized = checkpoint.proprio_standardizer.transform(
        np.asarray(proprio, dtype=np.float32).reshape(1, -1)
    )
    with torch.no_grad():
        action = checkpoint.model(
            images_to_tensor(image[None, ...]),
            torch.tensor([int(instruction_id)], dtype=torch.long),
            torch.from_numpy(normalized),
        )
    return action.squeeze(0).numpy().astype(np.float32)


def embedding_action(
    checkpoint: VLACheckpoint,
    visual: np.ndarray,
    text: np.ndarray,
    proprio: np.ndarray,
) -> np.ndarray:
    if not isinstance(checkpoint.model, ClipActionModel):
        raise TypeError("embedding_action expects a CLIP action-head checkpoint")
    checkpoint.model.eval()
    normalized = checkpoint.proprio_standardizer.transform(
        np.asarray(proprio, dtype=np.float32).reshape(1, -1)
    )
    with torch.no_grad():
        action = checkpoint.model(
            torch.from_numpy(np.asarray(visual, dtype=np.float32).reshape(1, -1)),
            torch.from_numpy(np.asarray(text, dtype=np.float32).reshape(1, -1)),
            torch.from_numpy(normalized),
        )
    return action.squeeze(0).numpy().astype(np.float32)
