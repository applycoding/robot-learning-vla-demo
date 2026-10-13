"""Proximal policy optimization baseline on the privileged state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from simreach.env import PlanarReachEnv


class EpisodeReturnCallback(BaseCallback):
    """Record undiscounted episode returns from the vectorized rollout."""

    def __init__(self) -> None:
        super().__init__()
        self.returns: list[float] = []
        self._current = 0.0

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals["dones"]).reshape(-1)
        self._current += float(rewards[0])
        if bool(dones[0]):
            self.returns.append(self._current)
            self._current = 0.0
        return True


def train_ppo(
    timesteps: int,
    seed: int = 0,
    path: str | Path | None = None,
) -> tuple[PPO, list[float]]:
    env = PlanarReachEnv()
    n_steps = 256 if timesteps < 2048 else 512
    batch_size = 64 if n_steps <= 256 else 128
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    callback = EpisodeReturnCallback()
    model.learn(total_timesteps=int(timesteps), callback=callback, progress_bar=False)
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
    return model, callback.returns


def load_ppo(path: str | Path) -> PPO:
    return PPO.load(str(path), device="cpu")


def ppo_action(model: PPO, obs: np.ndarray) -> np.ndarray:
    action, _state = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
    return np.asarray(action, dtype=np.float32).reshape(2)
