"""Gymnasium environment: a 2-link planar arm reaching a named target."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simreach.constants import (
    ACTION_DIM,
    DT,
    IMAGE_SIZE,
    INSTRUCTIONS,
    MAX_JOINT_VEL,
    MAX_STEPS,
    OBS_DIM,
    SUCCESS_THRESHOLD,
)
from simreach.features import state_features
from simreach.kinematics import forward_kinematics, integrate
from simreach.render import render_scene
from simreach.task import distance_to_goal, reward_from_state


class PlanarReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Reach the red or blue target named by the language instruction.

    The observation is a privileged state vector (joints, end effector, both
    target positions, and an instruction one-hot). Camera images are available
    through :meth:`render` and are what the vision-language policy consumes.

    Action is a normalized joint-velocity command in ``[-1, 1]``.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        image_size: int = IMAGE_SIZE,
        max_steps: int = MAX_STEPS,
        success_threshold: float = SUCCESS_THRESHOLD,
    ) -> None:
        super().__init__()
        if render_mode != "rgb_array":
            raise ValueError("Only render_mode='rgb_array' is supported")
        self.render_mode = render_mode
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.success_threshold = float(success_threshold)
        self.dt = DT
        self.max_joint_vel = MAX_JOINT_VEL
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-8.0, high=8.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.q = np.zeros(2, dtype=np.float64)
        self.q_dot = np.zeros(2, dtype=np.float64)
        self.red = np.zeros(2, dtype=np.float64)
        self.blue = np.zeros(2, dtype=np.float64)
        self.instruction_id = 0
        self.steps = 0

    @property
    def instruction(self) -> str:
        return INSTRUCTIONS[int(self.instruction_id)]

    @property
    def ee(self) -> np.ndarray:
        return forward_kinematics(self.q)

    @property
    def goal(self) -> np.ndarray:
        if self.instruction_id == 0:
            return self.red.copy()
        return self.blue.copy()

    def _sample_target(self) -> np.ndarray:
        radius = float(self.np_random.uniform(0.55, 1.65))
        angle = float(self.np_random.uniform(-np.pi, np.pi))
        return np.array(
            [radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64
        )

    def _sample_targets(self) -> tuple[np.ndarray, np.ndarray]:
        red = self._sample_target()
        blue = self._sample_target()
        for _ in range(100):
            if np.linalg.norm(red - blue) >= 0.75:
                return red, blue
            blue = self._sample_target()
        return red, blue

    def _sample_q(self, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        q = np.array([0.4, 1.2], dtype=np.float64)
        for _ in range(40):
            q = np.array(
                [
                    float(self.np_random.uniform(-2.2, 2.2)),
                    float(self.np_random.uniform(0.7, 2.2))
                    * (1.0 if self.np_random.random() < 0.5 else -1.0),
                ],
                dtype=np.float64,
            )
            ee = forward_kinematics(q)
            if min(np.linalg.norm(ee - red), np.linalg.norm(ee - blue)) > 0.5:
                return q
        return q

    def _apply_options(self, options: dict[str, object] | None) -> None:
        if not options:
            return
        if "q" in options:
            self.q = np.asarray(options["q"], dtype=np.float64).reshape(2).copy()
        if "red" in options:
            self.red = np.asarray(options["red"], dtype=np.float64).reshape(2).copy()
        if "blue" in options:
            self.blue = np.asarray(options["blue"], dtype=np.float64).reshape(2).copy()
        if "instruction_id" in options:
            self.instruction_id = int(options["instruction_id"])  # type: ignore[arg-type]

    def _obs(self) -> np.ndarray:
        return state_features(self.q, self.q_dot, self.red, self.blue, self.instruction_id)

    def _info(self) -> dict[str, object]:
        dist = distance_to_goal(self.q, self.red, self.blue, self.instruction_id)
        return {
            "distance": dist,
            "success": bool(dist < self.success_threshold),
            "instruction": self.instruction,
            "instruction_id": int(self.instruction_id),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self.red, self.blue = self._sample_targets()
        self.q = self._sample_q(self.red, self.blue)
        self.q_dot = np.zeros(2, dtype=np.float64)
        self.instruction_id = int(self.np_random.integers(0, 2))
        self.steps = 0
        self._apply_options(options)
        return self._obs(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.q, self.q_dot = integrate(self.q, action, self.dt, self.max_joint_vel)
        self.steps += 1
        reward, _dist, success = reward_from_state(
            self.q,
            self.red,
            self.blue,
            self.instruction_id,
            success_threshold=self.success_threshold,
        )
        terminated = bool(success)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        return self._obs(), float(reward), terminated, truncated, self._info()

    def render(self) -> np.ndarray:
        return render_scene(
            self.q,
            self.red,
            self.blue,
            self.instruction_id,
            self.image_size,
            annotate=False,
        )
