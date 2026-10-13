"""Planar 2-link kinematics and the velocity-control integrator."""

from __future__ import annotations

import numpy as np

from simreach.constants import DT, L1, L2, MAX_JOINT_VEL


def forward_kinematics(
    q: np.ndarray,
    l1: float = L1,
    l2: float = L2,
) -> np.ndarray:
    """End-effector position of a 2-link arm. ``q`` is ``[q1, q2]`` in radians."""
    q1 = float(q[0])
    q2 = float(q[1])
    ee_x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    ee_y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([ee_x, ee_y], dtype=np.float64)


def jacobian(
    q: np.ndarray,
    l1: float = L1,
    l2: float = L2,
) -> np.ndarray:
    """Analytic Jacobian mapping joint velocity (rad/s) to end-effector velocity."""
    q1 = float(q[0])
    q2 = float(q[1])
    s1 = np.sin(q1)
    c1 = np.cos(q1)
    s12 = np.sin(q1 + q2)
    c12 = np.cos(q1 + q2)
    return np.array(
        [
            [-l1 * s1 - l2 * s12, -l2 * s12],
            [l1 * c1 + l2 * c12, l2 * c12],
        ],
        dtype=np.float64,
    )


def integrate(
    q: np.ndarray,
    action: np.ndarray,
    dt: float = DT,
    max_joint_vel: float = MAX_JOINT_VEL,
) -> tuple[np.ndarray, np.ndarray]:
    """One Euler step of velocity control.

    ``action`` is a normalized joint-velocity command in ``[-1, 1]``.
    Returns ``(q_next, q_dot)`` with ``q_dot`` in rad/s.
    """
    command = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
    q_dot = command * float(max_joint_vel)
    q_next = np.asarray(q, dtype=np.float64).reshape(2) + q_dot * float(dt)
    return q_next, q_dot
