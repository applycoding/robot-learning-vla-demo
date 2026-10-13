"""Top-down RGB renderer for the planar arm.

Training images contain only the scene. Instruction text is drawn on qualitative
GIF frames so a vision policy cannot read the command off the pixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from simreach.constants import INSTRUCTIONS, L1
from simreach.kinematics import forward_kinematics

_BG = (15, 23, 42)
_GRID = (30, 41, 59)
_BASE = (148, 163, 184)
_LINK1 = (226, 232, 240)
_LINK2 = (148, 163, 184)
_EE = (248, 250, 252)
_RED = (239, 68, 68)
_BLUE = (37, 99, 235)
_RING = (248, 250, 252)
_TEXT = (226, 232, 240)

_WORLD_SPAN = 4.4
_WORLD_ORIGIN = 2.2


def _to_px(xy: np.ndarray, size: int) -> tuple[float, float]:
    x = float(xy[0])
    y = float(xy[1])
    px = (x + _WORLD_ORIGIN) / _WORLD_SPAN * (size - 1)
    py = (_WORLD_ORIGIN - y) / _WORLD_SPAN * (size - 1)
    return px, py


def _disk(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius_px: float,
    fill: tuple[int, int, int],
) -> None:
    cx, cy = center
    r = float(radius_px)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)


def render_scene(
    q: np.ndarray,
    red: np.ndarray,
    blue: np.ndarray,
    instruction_id: int,
    image_size: int,
    annotate: bool = False,
    step: int | None = None,
    success: bool | None = None,
) -> np.ndarray:
    """Render one RGB frame as ``uint8`` array of shape ``(H, W, 3)``."""
    size = int(image_size)
    if size < 16:
        raise ValueError("image_size must be at least 16")
    image = Image.new("RGB", (size, size), _BG)
    draw = ImageDraw.Draw(image)

    for tick in np.linspace(-2.0, 2.0, 9):
        draw.line(
            [_to_px(np.array([tick, -2.2]), size), _to_px(np.array([tick, 2.2]), size)],
            fill=_GRID,
            width=1,
        )
        draw.line(
            [_to_px(np.array([-2.2, tick]), size), _to_px(np.array([2.2, tick]), size)],
            fill=_GRID,
            width=1,
        )

    scale = size / _WORLD_SPAN
    target_r = max(8.0, 0.28 * scale)
    q = np.asarray(q, dtype=np.float64).reshape(2)
    elbow = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])], dtype=np.float64)
    ee = forward_kinematics(q)
    base = np.zeros(2, dtype=np.float64)

    red_px = _to_px(red, size)
    blue_px = _to_px(blue, size)
    _disk(draw, red_px, target_r, _RED)
    _disk(draw, blue_px, target_r, _BLUE)
    # The ring is a qualitative highlight for GIFs. Training frames omit it so the
    # instruction is not painted into the pixels the vision policy trains on.
    if annotate:
        ring_center = red_px if int(instruction_id) == 0 else blue_px
        ring_r = target_r + max(2.0, size / 48.0)
        draw.ellipse(
            [
                ring_center[0] - ring_r,
                ring_center[1] - ring_r,
                ring_center[0] + ring_r,
                ring_center[1] + ring_r,
            ],
            outline=_RING,
            width=max(2, size // 80),
        )

    width = max(2, size // 48)
    draw.line([_to_px(base, size), _to_px(elbow, size)], fill=_LINK1, width=width)
    draw.line([_to_px(elbow, size), _to_px(ee, size)], fill=_LINK2, width=width)
    joint_r = max(2.5, 0.07 * scale)
    _disk(draw, _to_px(base, size), joint_r + 1.0, _BASE)
    _disk(draw, _to_px(elbow, size), joint_r, _LINK1)
    _disk(draw, _to_px(ee, size), joint_r + 0.5, _EE)

    if annotate and size >= 96:
        bar_h = max(16, size // 12)
        draw.rectangle([0, 0, size, bar_h], fill=_BG)
        label = INSTRUCTIONS[int(instruction_id)]
        if step is not None:
            label = f"{label}   t={step}"
        if success:
            label = f"{label}   success"
        draw.text((6, 3), label, fill=_TEXT)

    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
