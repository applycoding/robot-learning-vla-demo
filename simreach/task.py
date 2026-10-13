"""Goal selection and the shaped reach reward."""

from __future__ import annotations

import numpy as np

from simreach.constants import SUCCESS_BONUS, SUCCESS_THRESHOLD
from simreach.kinematics import forward_kinematics


def instructed_goal(
    red: np.ndarray,
    blue: np.ndarray,
    instruction_id: int,
) -> np.ndarray:
    if instruction_id == 0:
        return np.asarray(red, dtype=np.float64)
    if instruction_id == 1:
        return np.asarray(blue, dtype=np.float64)
    raise ValueError(f"instruction_id must be 0 or 1, got {instruction_id}")


def tracked_target(
    red: np.ndarray,
    blue: np.ndarray,
    tracked_color: int,
) -> np.ndarray:
    return instructed_goal(red, blue, tracked_color)


def other_color(instruction_id: int) -> int:
    if instruction_id == 0:
        return 1
    if instruction_id == 1:
        return 0
    raise ValueError(f"instruction_id must be 0 or 1, got {instruction_id}")


def distance_to_goal(
    q: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    instruction_id: int,
) -> float:
    ee = forward_kinematics(q)
    goal = instructed_goal(red, blue, instruction_id)
    return float(np.linalg.norm(ee - goal))


def reward_from_state(
    q: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    instruction_id: int,
    success_threshold: float = SUCCESS_THRESHOLD,
    success_bonus: float = SUCCESS_BONUS,
) -> tuple[float, float, bool]:
    """Return ``(reward, distance, success)`` for a joint state.

    Reward is ``exp(-3 * distance)``, plus ``success_bonus`` inside the success ball.
    """
    dist = distance_to_goal(q, red, blue, instruction_id)
    success = dist < success_threshold
    # Dense hill plus a terminal bonus. The hill gives PPO a direction; the bonus
    # makes actually reaching the target worth more than hovering nearby.
    reward = float(np.exp(-3.0 * dist))
    if success:
        reward += float(success_bonus)
    return reward, float(dist), bool(success)
