"""Behavior cloning, VLA action head, PPO smoke test, and the quick benchmark."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from simreach.bc import bc_action, train_bc
from simreach.benchmark import BenchmarkConfig, run_benchmark
from simreach.clip_encoder import ClipEncoder
from simreach.constants import Quality
from simreach.dataset import generate_episode
from simreach.env import PlanarReachEnv
from simreach.evaluate import evaluate_policy
from simreach.expert import expert_action
from simreach.features import proprio_features
from simreach.ppo_train import ppo_action, train_ppo
from simreach.vla import TinyVLA, images_to_tensor, tiny_vla_action, train_embedding_head, train_tiny_vla


class _StubClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=4)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del input_ids, attention_mask
        assert pixel_values is not None
        batch = int(pixel_values.shape[0])
        return SimpleNamespace(
            image_embeds=torch.ones(batch, 4),
            text_embeds=torch.full((batch, 4), 2.0),
        )


class _StubProcessor:
    def __call__(
        self,
        text: list[str],
        images: list[object],
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, torch.Tensor]:
        del images, return_tensors, padding
        count = len(text)
        return {
            "pixel_values": torch.zeros(count, 3, 8, 8),
            "input_ids": torch.ones(count, 3, dtype=torch.long),
            "attention_mask": torch.ones(count, 3, dtype=torch.long),
        }


def test_behavior_cloning_loss_decreases() -> None:
    episodes = [generate_episode(100 + i, Quality.GOLD, image_size=32) for i in range(6)]
    checkpoint = train_bc(episodes, epochs=6, seed=0, hidden=32, weight_decay=0.0, obs_noise=0.0)
    assert checkpoint.losses[-1] < checkpoint.losses[0]
    env = PlanarReachEnv(image_size=32)
    _obs, _info = env.reset(seed=0)
    action = bc_action(checkpoint, env._obs())
    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0 + 1e-5)


def test_tiny_vla_trains_and_respects_action_bounds() -> None:
    episodes = [generate_episode(200 + i, Quality.GOLD, image_size=32) for i in range(4)]
    checkpoint = train_tiny_vla(episodes, epochs=2, seed=0, hidden=32)
    assert checkpoint.losses[-1] <= checkpoint.losses[0] + 1e-6
    env = PlanarReachEnv(image_size=32)
    env.reset(seed=1)
    action = tiny_vla_action(
        checkpoint,
        env.render(),
        env.instruction_id,
        proprio_features(env.q, env.q_dot),
    )
    assert action.shape == (2,)
    assert float(np.max(np.abs(action))) <= 1.0 + 1e-5
    batch = images_to_tensor(np.stack([env.render(), env.render()]))
    assert batch.shape[1:] == (3, 32, 32)
    model = checkpoint.model
    assert isinstance(model, TinyVLA)


def test_clip_encoder_stub_and_action_head() -> None:
    encoder = ClipEncoder(model=_StubClip(), processor=_StubProcessor())
    images = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    visual, text = encoder.embed_numpy(images, ["reach the red target"] * 3, batch_size=2)
    assert visual.shape == (3, 4)
    assert np.allclose(text, 2.0)
    proprio = np.zeros((24, 6), dtype=np.float32)
    actions = np.tanh(np.linspace(-1, 1, 48).reshape(24, 2)).astype(np.float32)
    visual_rows = np.ones((24, 4), dtype=np.float32)
    text_rows = np.ones((24, 4), dtype=np.float32)
    checkpoint = train_embedding_head(
        visual_rows, text_rows, proprio, actions, epochs=3, seed=0, hidden=16
    )
    assert checkpoint.kind == "clip"
    assert checkpoint.losses[-1] < checkpoint.losses[0]


def test_ppo_smoke_predicts_action() -> None:
    model, returns = train_ppo(256, seed=0)
    assert returns is not None
    env = PlanarReachEnv()
    obs, _info = env.reset(seed=0)
    action = ppo_action(model, obs)
    assert action.shape == (2,)


def test_expert_beats_random_on_shared_seeds() -> None:
    def _expert(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
        del obs
        return expert_action(env.q, env.goal)

    rng = np.random.default_rng(0)

    def _random(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
        del env, obs
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    expert = evaluate_policy("expert", "scripted", _expert, [0, 1, 2, 3], 32)
    random = evaluate_policy("random", "none", _random, [0, 1, 2, 3], 32)
    assert expert.success_rate == 1.0
    assert expert.success_rate > random.success_rate


def test_quick_benchmark_writes_artifacts(tmp_path: Path) -> None:
    metrics = run_benchmark(BenchmarkConfig.quick(), tmp_path)
    assert metrics["validation_passed"] is True
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "success_rates.png").is_file()
    assert (tmp_path / "data_quality.png").is_file()
    assert (tmp_path / "learning_curves.png").is_file()
    assert (tmp_path / "expert.gif").is_file()
    assert (tmp_path / "bc.gif").is_file()
    assert (tmp_path / "validation.json").is_file()
    methods = {row["method"] for row in metrics["results"]}  # type: ignore[index]
    assert {"expert", "behavior cloning", "vla", "ppo", "random"} <= methods
