"""Flat observation vectors for state-based policies and VLA proprioception."""

from __future__ import annotations

import numpy as np

from simreach.constants import OBS_DIM, PROPRIO_DIM
from simreach.kinematics import forward_kinematics
from simreach.task import instructed_goal


def state_features(
    q: np.ndarray,
    q_dot: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    instruction_id: int,
) -> np.ndarray:
    """Privileged state: joints, end effector, instructed goal offset, one-hot."""
    q = np.asarray(q, dtype=np.float64).reshape(2)
    q_dot = np.asarray(q_dot, dtype=np.float64).reshape(2)
    ee = forward_kinematics(q)
    goal = instructed_goal(red, blue, int(instruction_id))
    instruction = np.zeros(2, dtype=np.float64)
    instruction[int(instruction_id)] = 1.0
    # The instructed target is expressed relative to the end effector so behavior
    # cloning learns a goal-conditioned reach rather than memorizing coordinates.
    vec = np.concatenate(
        [
            [np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1])],
            q_dot,
            ee,
            goal - ee,
            instruction,
        ]
    )
    if vec.shape != (OBS_DIM,):
        raise RuntimeError(f"expected state dim {OBS_DIM}, got {vec.shape}")
    return vec.astype(np.float32)


def proprio_features(q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
    """Joint features available to a vision policy. Target locations are omitted."""
    q = np.asarray(q, dtype=np.float64).reshape(2)
    q_dot = np.asarray(q_dot, dtype=np.float64).reshape(2)
    vec = np.array(
        [np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1]), q_dot[0], q_dot[1]],
        dtype=np.float32,
    )
    if vec.shape != (PROPRIO_DIM,):
        raise RuntimeError(f"expected proprio dim {PROPRIO_DIM}, got {vec.shape}")
    return vec
