"""Physical, observation, and dataset constants shared across the project."""

from __future__ import annotations

from enum import Enum

L1: float = 1.0
L2: float = 1.0
DT: float = 0.1
MAX_JOINT_VEL: float = 2.0
MAX_STEPS: int = 50
SUCCESS_THRESHOLD: float = 0.12
SUCCESS_BONUS: float = 5.0
IMAGE_SIZE: int = 64
SCHEMA_VERSION: str = "1.0.0"
FPS: int = 10

INSTRUCTIONS: tuple[str, str] = (
    "reach the red target",
    "reach the blue target",
)

OBS_DIM: int = 12
PROPRIO_DIM: int = 6
ACTION_DIM: int = 2

STATE_NAMES: tuple[str, ...] = (
    "sin_q1",
    "cos_q1",
    "sin_q2",
    "cos_q2",
    "q1_dot",
    "q2_dot",
    "ee_x",
    "ee_y",
    "goal_rel_x",
    "goal_rel_y",
    "instr_red",
    "instr_blue",
)

PROPRIO_NAMES: tuple[str, ...] = STATE_NAMES[:PROPRIO_DIM]


class Quality(str, Enum):
    """Demonstration tier. Values are the on-disk annotation strings."""

    GOLD = "gold"
    NOISY = "noisy"
    MISLABELED = "mislabeled"
