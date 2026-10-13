"""Scripted reaching expert and the action corruptions used for data tiers."""

from __future__ import annotations

import numpy as np

from simreach.constants import MAX_JOINT_VEL, Quality
from simreach.kinematics import forward_kinematics, jacobian
from simreach.util import assert_never

NOISE_STD: float = 0.85
RANDOM_ACTION_PROB: float = 0.45
# Noisy commands keep only a fraction of the expert so the corruption is not
# averaged away by behavior cloning.
NOISY_EXPERT_SCALE: float = 0.35
_EE_GAIN: float = 2.5
_EE_MAX_SPEED: float = 1.6
_DAMPING: float = 0.05


def expert_action(
    q: np.ndarray,
    goal: np.ndarray,
    max_joint_vel: float = MAX_JOINT_VEL,
) -> np.ndarray:
    """Damped least-squares joint velocity toward ``goal``, normalized to ``[-1, 1]``."""
    ee = forward_kinematics(q)
    error = np.asarray(goal, dtype=np.float64).reshape(2) - ee
    dist = float(np.linalg.norm(error))
    if dist < 1e-8:
        return np.zeros(2, dtype=np.float32)
    velocity = error * _EE_GAIN
    speed = float(np.linalg.norm(velocity))
    if speed > _EE_MAX_SPEED:
        velocity = velocity / speed * _EE_MAX_SPEED
    jac = jacobian(q)
    gram = jac @ jac.T + (_DAMPING**2) * np.eye(2)
    q_dot = jac.T @ np.linalg.solve(gram, velocity)
    peak = float(np.max(np.abs(q_dot)))
    limit = float(max_joint_vel)
    if peak > limit:
        q_dot = q_dot * (limit / peak)
    action = np.clip(q_dot / limit, -1.0, 1.0)
    return action.astype(np.float32)


def corrupt_action(action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Noisy-tier action: mixture of Gaussian jitter and fully random commands."""
    if float(rng.random()) < RANDOM_ACTION_PROB:
        noisy = rng.uniform(-1.0, 1.0, size=2)
    else:
        base = NOISY_EXPERT_SCALE * np.asarray(action, dtype=np.float64)
        noisy = base + rng.normal(0.0, NOISE_STD, size=2)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def action_for_quality(
    q: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    tracked_color: int,
    quality: Quality,
    rng: np.random.Generator,
) -> np.ndarray:
    goal = red if int(tracked_color) == 0 else blue
    action = expert_action(q, goal)
    match quality:
        case Quality.GOLD | Quality.MISLABELED:
            return action
        case Quality.NOISY:
            return corrupt_action(action, rng)
        case _:
            assert_never(quality)
