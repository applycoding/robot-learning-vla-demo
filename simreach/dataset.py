"""HDF5 demonstration dataset, LeRobot-style sidecar metadata, and episode I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from simreach.constants import (
    ACTION_DIM,
    DT,
    FPS,
    IMAGE_SIZE,
    INSTRUCTIONS,
    L1,
    L2,
    MAX_JOINT_VEL,
    OBS_DIM,
    PROPRIO_DIM,
    SCHEMA_VERSION,
    STATE_NAMES,
    SUCCESS_BONUS,
    SUCCESS_THRESHOLD,
    Quality,
)
from simreach.env import PlanarReachEnv
from simreach.expert import action_for_quality
from simreach.features import proprio_features, state_features
from simreach.task import other_color, reward_from_state
from simreach.util import assert_never

TRANSITION_DATASETS = ("action", "reward")
STATE_DATASETS = ("q", "q_dot", "image")


@dataclass
class Episode:
    """One demonstration.

    ``q``, ``q_dot``, and ``image`` have length ``T + 1`` (the terminal state is
    included). ``action`` and ``reward`` have length ``T``.
    """

    q: np.ndarray
    q_dot: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    image: np.ndarray
    target_red: np.ndarray
    target_blue: np.ndarray
    instruction_id: int
    instruction: str
    quality: Quality
    seed: int
    success: bool
    tracked_color: int

    @property
    def num_transitions(self) -> int:
        return int(self.action.shape[0])


def generate_episode(
    seed: int,
    quality: Quality,
    image_size: int = IMAGE_SIZE,
) -> Episode:
    """Roll out the scripted expert under a quality tier."""
    env = PlanarReachEnv(image_size=image_size)
    _obs, _info = env.reset(seed=seed)
    match quality:
        case Quality.GOLD | Quality.NOISY:
            tracked = int(env.instruction_id)
        case Quality.MISLABELED:
            tracked = other_color(int(env.instruction_id))
        case _:
            assert_never(quality)
    noise_rng = np.random.default_rng(seed + 7919)
    qs = [env.q.copy()]
    qdots = [env.q_dot.copy()]
    images = [env.render()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    for _ in range(env.max_steps):
        action = action_for_quality(
            env.q, env.red, env.blue, tracked, quality, noise_rng
        )
        _obs, reward, terminated, truncated, _info = env.step(action)
        actions.append(np.asarray(action, dtype=np.float32).copy())
        rewards.append(float(reward))
        qs.append(env.q.copy())
        qdots.append(env.q_dot.copy())
        images.append(env.render())
        if terminated or truncated:
            break
    success = False
    for q in qs[1:]:
        _reward, _dist, hit = reward_from_state(
            q, env.red, env.blue, env.instruction_id
        )
        success = success or hit
    return Episode(
        q=np.stack(qs).astype(np.float32),
        q_dot=np.stack(qdots).astype(np.float32),
        action=np.stack(actions).astype(np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        image=np.stack(images).astype(np.uint8),
        target_red=env.red.astype(np.float32),
        target_blue=env.blue.astype(np.float32),
        instruction_id=int(env.instruction_id),
        instruction=env.instruction,
        quality=quality,
        seed=int(seed),
        success=bool(success),
        tracked_color=int(tracked),
    )


def generate_dataset(
    n_gold: int,
    n_noisy: int,
    n_mislabeled: int,
    seed: int = 0,
    image_size: int = IMAGE_SIZE,
) -> list[Episode]:
    """Generate non-overlapping episodes. Eval seeds ``0..N`` are left unused."""
    plan = (
        (Quality.GOLD, n_gold),
        (Quality.NOISY, n_noisy),
        (Quality.MISLABELED, n_mislabeled),
    )
    episodes: list[Episode] = []
    cursor = 10_000 + int(seed) * 100_000
    for quality, count in plan:
        for _ in range(count):
            episodes.append(generate_episode(cursor, quality, image_size=image_size))
            cursor += 1
    return episodes


def _as_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def write_dataset(path: str | Path, episodes: list[Episode], image_size: int) -> Path:
    """Write ``reach`` demonstrations to HDF5 and a LeRobot-style ``meta/info.json``."""
    h5_path = Path(path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["dt"] = DT
        handle.attrs["max_joint_vel"] = MAX_JOINT_VEL
        handle.attrs["l1"] = L1
        handle.attrs["l2"] = L2
        handle.attrs["success_threshold"] = SUCCESS_THRESHOLD
        handle.attrs["success_bonus"] = SUCCESS_BONUS
        handle.attrs["image_size"] = int(image_size)
        handle.attrs["fps"] = FPS
        handle.create_dataset(
            "instruction_vocab",
            data=np.array(INSTRUCTIONS, dtype=object),
            dtype=string_dtype,
        )
        root = handle.create_group("episodes")
        for index, episode in enumerate(episodes):
            group = root.create_group(f"{index:06d}")
            group.create_dataset("q", data=episode.q, compression="gzip")
            group.create_dataset("q_dot", data=episode.q_dot, compression="gzip")
            group.create_dataset("action", data=episode.action, compression="gzip")
            group.create_dataset("reward", data=episode.reward, compression="gzip")
            group.create_dataset(
                "image", data=episode.image, compression="gzip", compression_opts=4
            )
            group.create_dataset("target_red", data=episode.target_red)
            group.create_dataset("target_blue", data=episode.target_blue)
            group.attrs["instruction_id"] = int(episode.instruction_id)
            group.attrs["instruction"] = episode.instruction
            group.attrs["quality"] = episode.quality.value
            group.attrs["seed"] = int(episode.seed)
            group.attrs["success"] = bool(episode.success)
            group.attrs["tracked_color"] = int(episode.tracked_color)
            group.attrs["episode_return"] = float(np.sum(episode.reward))
    info = lerobot_info(episodes, image_size)
    meta_dir = h5_path.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info_path = meta_dir / "info.json"
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return h5_path


def lerobot_info(episodes: list[Episode], image_size: int) -> dict[str, object]:
    """Sidecar metadata using LeRobot-style feature names.

    The episode payloads live in HDF5 rather than parquet plus mp4. Field names
    match the LeRobot convention so a later export is mechanical.
    """
    total_frames = int(sum(ep.num_transitions for ep in episodes))
    counts = {quality.value: 0 for quality in Quality}
    for episode in episodes:
        counts[episode.quality.value] += 1
    return {
        "codebase_version": "simreach-v1",
        "schema_version": SCHEMA_VERSION,
        "robot_type": "planar_2link",
        "fps": FPS,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "quality_counts": counts,
        "storage": {
            "format": "hdf5",
            "episodes_group": "/episodes/{index:06d}",
            "state_length": "T+1",
            "action_length": "T",
        },
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [OBS_DIM],
                "names": list(STATE_NAMES),
            },
            "observation.proprio": {
                "dtype": "float32",
                "shape": [PROPRIO_DIM],
                "names": list(STATE_NAMES[:PROPRIO_DIM]),
            },
            "observation.image": {
                "dtype": "uint8",
                "shape": [int(image_size), int(image_size), 3],
                "names": ["height", "width", "channel"],
            },
            "observation.language": {
                "dtype": "string",
                "vocab": list(INSTRUCTIONS),
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ["q1_vel_norm", "q2_vel_norm"],
                "range": [-1.0, 1.0],
            },
        },
        "simulator": {
            "dt": DT,
            "max_joint_vel": MAX_JOINT_VEL,
            "l1": L1,
            "l2": L2,
            "success_threshold": SUCCESS_THRESHOLD,
            "success_bonus": SUCCESS_BONUS,
        },
    }


def load_episodes(path: str | Path, load_images: bool = True) -> list[Episode]:
    episodes: list[Episode] = []
    with h5py.File(path, "r") as handle:
        root = handle["episodes"]
        for key in sorted(root.keys()):
            group = root[key]
            image = (
                np.asarray(group["image"], dtype=np.uint8)
                if load_images
                else np.zeros((0, 0, 0, 0), dtype=np.uint8)
            )
            episodes.append(
                Episode(
                    q=np.asarray(group["q"], dtype=np.float32),
                    q_dot=np.asarray(group["q_dot"], dtype=np.float32),
                    action=np.asarray(group["action"], dtype=np.float32),
                    reward=np.asarray(group["reward"], dtype=np.float32),
                    image=image,
                    target_red=np.asarray(group["target_red"], dtype=np.float32),
                    target_blue=np.asarray(group["target_blue"], dtype=np.float32),
                    instruction_id=int(group.attrs["instruction_id"]),
                    instruction=_as_str(group.attrs["instruction"]),
                    quality=Quality(_as_str(group.attrs["quality"])),
                    seed=int(group.attrs["seed"]),
                    success=bool(group.attrs["success"]),
                    tracked_color=int(group.attrs["tracked_color"]),
                )
            )
    return episodes


def filter_quality(episodes: list[Episode], quality: Quality) -> list[Episode]:
    return [episode for episode in episodes if episode.quality is quality or episode.quality == quality]


def iter_transitions(
    episodes: list[Episode],
) -> list[tuple[Episode, int]]:
    pairs: list[tuple[Episode, int]] = []
    for episode in episodes:
        for index in range(episode.num_transitions):
            pairs.append((episode, index))
    return pairs


def stack_bc_batch(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    """Stack privileged state observations and actions for behavior cloning."""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode, index in iter_transitions(episodes):
        observations.append(
            state_features(
                episode.q[index],
                episode.q_dot[index],
                episode.target_red,
                episode.target_blue,
                episode.instruction_id,
            )
        )
        actions.append(episode.action[index])
    if not observations:
        raise ValueError("no transitions to stack")
    return np.stack(observations).astype(np.float32), np.stack(actions).astype(np.float32)


def stack_vla_batch(
    episodes: list[Episode],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack images, instruction ids, proprioception, and actions."""
    images: list[np.ndarray] = []
    instructions: list[int] = []
    proprios: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for episode, index in iter_transitions(episodes):
        if episode.image.ndim != 4:
            raise ValueError("VLA training requires episode images")
        images.append(episode.image[index])
        instructions.append(int(episode.instruction_id))
        proprios.append(proprio_features(episode.q[index], episode.q_dot[index]))
        actions.append(episode.action[index])
    if not images:
        raise ValueError("no transitions to stack")
    return (
        np.stack(images).astype(np.uint8),
        np.asarray(instructions, dtype=np.int64),
        np.stack(proprios).astype(np.float32),
        np.stack(actions).astype(np.float32),
    )
