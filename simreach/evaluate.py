"""Shared evaluation: success rate, return, and final distance on fixed seeds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from simreach.env import PlanarReachEnv
from simreach.render import render_scene

ActionFn = Callable[[PlanarReachEnv, np.ndarray], np.ndarray]


@dataclass
class EpisodeResult:
    success: bool
    episode_return: float
    final_distance: float
    steps: int
    frames: list[np.ndarray]


@dataclass
class EvalSummary:
    method: str
    data: str
    success_rate: float
    mean_return: float
    mean_final_distance: float
    n: int
    episode_success: list[bool]
    seeds: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "data": self.data,
            "success_rate": self.success_rate,
            "mean_return": self.mean_return,
            "mean_final_distance": self.mean_final_distance,
            "n": self.n,
        }


def run_episode(
    env: PlanarReachEnv,
    act: ActionFn,
    seed: int,
    record_size: int | None = None,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    total = 0.0
    success = bool(info["success"])
    final_distance = float(info["distance"])
    frames: list[np.ndarray] = []
    if record_size is not None:
        frames.append(
            render_scene(
                env.q,
                env.red,
                env.blue,
                env.instruction_id,
                record_size,
                annotate=True,
                step=0,
                success=success,
            )
        )
    steps = 0
    for _ in range(env.max_steps):
        action = np.asarray(act(env, obs), dtype=np.float32).reshape(2)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        success = success or bool(info["success"])
        final_distance = float(info["distance"])
        if record_size is not None:
            frames.append(
                render_scene(
                    env.q,
                    env.red,
                    env.blue,
                    env.instruction_id,
                    record_size,
                    annotate=True,
                    step=steps,
                    success=success,
                )
            )
        if terminated or truncated:
            break
    return EpisodeResult(
        success=success,
        episode_return=total,
        final_distance=final_distance,
        steps=steps,
        frames=frames,
    )


def evaluate_policy(
    method: str,
    data: str,
    act: ActionFn,
    seeds: list[int],
    image_size: int,
) -> EvalSummary:
    env = PlanarReachEnv(image_size=image_size)
    results = [run_episode(env, act, seed) for seed in seeds]
    return EvalSummary(
        method=method,
        data=data,
        success_rate=float(np.mean([result.success for result in results])) if results else 0.0,
        mean_return=float(np.mean([result.episode_return for result in results])) if results else 0.0,
        mean_final_distance=float(np.mean([result.final_distance for result in results])) if results else 0.0,
        n=len(results),
        episode_success=[result.success for result in results],
        seeds=list(seeds),
    )


def markdown_table(summaries: list[EvalSummary]) -> str:
    lines = [
        "| Method | Data | Success | Mean return | Mean final distance |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.method} | {summary.data} | {100 * summary.success_rate:.1f}% | "
            f"{summary.mean_return:.2f} | {summary.mean_final_distance:.3f} |"
        )
    return "\n".join(lines)
