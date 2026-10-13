"""Dataset validation: schema, kinematics, labels, images, and tier standards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from simreach.constants import (
    DT,
    INSTRUCTIONS,
    MAX_JOINT_VEL,
    SCHEMA_VERSION,
    SUCCESS_THRESHOLD,
    Quality,
)
from simreach.dataset import Episode, load_episodes
from simreach.kinematics import forward_kinematics, integrate
from simreach.task import distance_to_goal, reward_from_state
from simreach.util import assert_never

# Gold demonstrations are smooth expert trajectories. This caps mean |Δaction|.
GOLD_MAX_MEAN_ACTION_DELTA = 0.45
_KINEMATIC_ATOL = 1e-4
_REWARD_ATOL = 1e-4
_IMAGE_MIN_STD = 2.0


@dataclass
class EpisodeAudit:
    index: int
    seed: int
    quality: str
    structural: list[str] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)
    kinematic_error: float = 0.0
    reward_error: float = 0.0
    success_recomputed: bool = False
    final_distance: float = 0.0
    smoothness: float = 0.0
    image_std: float = 0.0

    @property
    def structural_ok(self) -> bool:
        return not self.structural


@dataclass
class ValidationReport:
    schema_version: str
    num_episodes: int
    num_transitions: int
    structural_failures: int
    gold_episodes: int
    gold_standard_passes: int
    by_quality: dict[str, dict[str, float]]
    stats: dict[str, float]
    audits: list[EpisodeAudit]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        failing = [
            {
                "index": audit.index,
                "seed": audit.seed,
                "quality": audit.quality,
                "structural": audit.structural,
                "quality_issues": audit.quality_issues,
            }
            for audit in self.audits
            if audit.structural or audit.quality_issues
        ]
        return {
            "schema_version": self.schema_version,
            "num_episodes": self.num_episodes,
            "num_transitions": self.num_transitions,
            "structural_failures": self.structural_failures,
            "gold_episodes": self.gold_episodes,
            "gold_standard_passes": self.gold_standard_passes,
            "gold_standard_pass_rate": (
                self.gold_standard_passes / self.gold_episodes if self.gold_episodes else 1.0
            ),
            "by_quality": self.by_quality,
            "stats": self.stats,
            "passed": self.passed,
            "flagged_episodes": failing[:30],
        }


def _mean_action_delta(action: np.ndarray) -> float:
    if len(action) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(action, axis=0))))


def audit_episode(episode: Episode, index: int, image_size: int | None = None) -> EpisodeAudit:
    """Recompute ground truth and compare it to the stored annotation."""
    audit = EpisodeAudit(
        index=index,
        seed=int(episode.seed),
        quality=episode.quality.value,
    )
    transitions = episode.num_transitions
    if episode.q.ndim != 2 or episode.q.shape[1] != 2:
        audit.structural.append("q must have shape (T+1, 2)")
    if episode.q_dot.shape != episode.q.shape:
        audit.structural.append("q_dot shape must match q")
    if episode.action.shape != (transitions, 2):
        audit.structural.append("action must have shape (T, 2)")
    if episode.reward.shape != (transitions,):
        audit.structural.append("reward must have shape (T,)")
    if transitions < 1 or episode.q.shape[0] != transitions + 1:
        audit.structural.append("state length must be action length + 1")
    if episode.instruction not in INSTRUCTIONS:
        audit.structural.append(f"instruction {episode.instruction!r} is outside the vocab")
    if int(episode.instruction_id) not in (0, 1):
        audit.structural.append("instruction_id must be 0 or 1")
    elif episode.instruction != INSTRUCTIONS[int(episode.instruction_id)]:
        audit.structural.append("instruction string does not match instruction_id")
    if not np.isfinite(episode.q).all() or not np.isfinite(episode.action).all():
        audit.structural.append("non-finite state or action")
    if episode.action.size and (
        np.max(episode.action) > 1.0 + 1e-4 or np.min(episode.action) < -1.0 - 1e-4
    ):
        audit.structural.append("action outside [-1, 1]")

    images_present = episode.image.ndim == 4 and episode.image.shape[0] > 0
    if images_present:
        expected_h = image_size if image_size is not None else int(episode.image.shape[1])
        if episode.image.shape != (episode.q.shape[0], expected_h, expected_h, 3):
            audit.structural.append(
                f"image shape {episode.image.shape} does not match states and image_size"
            )
        if episode.image.dtype != np.uint8:
            audit.structural.append("image dtype must be uint8")
        audit.image_std = float(np.std(episode.image.astype(np.float32)))
        if audit.image_std < _IMAGE_MIN_STD:
            audit.structural.append("image is blank or nearly constant")
    else:
        audit.structural.append("image missing")

    if audit.structural:
        return audit

    max_kin = 0.0
    max_reward = 0.0
    recomputed_success = False
    for t in range(transitions):
        q_next, q_dot = integrate(
            episode.q[t].astype(np.float64),
            episode.action[t].astype(np.float64),
            DT,
            MAX_JOINT_VEL,
        )
        max_kin = max(max_kin, float(np.max(np.abs(q_next - episode.q[t + 1].astype(np.float64)))))
        max_kin = max(
            max_kin, float(np.max(np.abs(q_dot - episode.q_dot[t + 1].astype(np.float64))))
        )
        if t == 0:
            max_kin = max(max_kin, float(np.max(np.abs(episode.q_dot[0]))))
        reward, _dist, hit = reward_from_state(
            episode.q[t + 1].astype(np.float64),
            episode.target_red.astype(np.float64),
            episode.target_blue.astype(np.float64),
            int(episode.instruction_id),
        )
        max_reward = max(max_reward, abs(reward - float(episode.reward[t])))
        recomputed_success = recomputed_success or hit
    audit.kinematic_error = max_kin
    audit.reward_error = max_reward
    audit.success_recomputed = recomputed_success
    audit.smoothness = _mean_action_delta(episode.action.astype(np.float64))
    audit.final_distance = distance_to_goal(
        episode.q[-1].astype(np.float64),
        episode.target_red.astype(np.float64),
        episode.target_blue.astype(np.float64),
        int(episode.instruction_id),
    )
    if max_kin > _KINEMATIC_ATOL:
        audit.structural.append(f"kinematic mismatch {max_kin:.3e} exceeds {_KINEMATIC_ATOL}")
    if max_reward > _REWARD_ATOL:
        audit.structural.append(f"reward mismatch {max_reward:.3e} exceeds {_REWARD_ATOL}")
    if bool(episode.success) != recomputed_success:
        audit.structural.append(
            f"stored success={episode.success} but recomputed success={recomputed_success}"
        )

    ee_final = forward_kinematics(episode.q[-1].astype(np.float64))
    red = episode.target_red.astype(np.float64)
    blue = episode.target_blue.astype(np.float64)
    dist_red = float(np.linalg.norm(ee_final - red))
    dist_blue = float(np.linalg.norm(ee_final - blue))
    closer_to_tracked = dist_red < dist_blue if episode.tracked_color == 0 else dist_blue < dist_red

    match episode.quality:
        case Quality.GOLD:
            if episode.tracked_color != episode.instruction_id:
                audit.quality_issues.append("gold episode tracks a different color than the instruction")
            if not recomputed_success:
                audit.quality_issues.append("gold episode does not reach the instructed target")
            if audit.smoothness > GOLD_MAX_MEAN_ACTION_DELTA:
                audit.quality_issues.append(
                    f"gold smoothness {audit.smoothness:.3f} exceeds {GOLD_MAX_MEAN_ACTION_DELTA}"
                )
            if not closer_to_tracked:
                audit.quality_issues.append("gold final pose is closer to the other target")
        case Quality.NOISY:
            if episode.tracked_color != episode.instruction_id:
                audit.quality_issues.append("noisy episode must still be labeled with the tracked color")
        case Quality.MISLABELED:
            if episode.tracked_color == episode.instruction_id:
                audit.quality_issues.append("mislabeled episode does not actually disagree with the instruction")
            if closer_to_tracked is False and recomputed_success:
                audit.quality_issues.append("mislabeled trajectory reached the instructed target")
        case _:
            assert_never(episode.quality)
    return audit


def _quality_stats(audits: list[EpisodeAudit], episodes: list[Episode]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for quality in Quality:
        selected = [ep for ep, audit in zip(episodes, audits) if ep.quality is quality]
        selected_audits = [audit for audit in audits if audit.quality == quality.value]
        if not selected:
            summary[quality.value] = {
                "episodes": 0,
                "success_rate": 0.0,
                "mean_length": 0.0,
                "mean_smoothness": 0.0,
                "structural_failures": 0,
                "quality_issues": 0,
            }
            continue
        summary[quality.value] = {
            "episodes": float(len(selected)),
            "success_rate": float(np.mean([ep.success for ep in selected])),
            "mean_length": float(np.mean([ep.num_transitions for ep in selected])),
            "mean_smoothness": float(np.mean([audit.smoothness for audit in selected_audits])),
            "structural_failures": float(sum(not audit.structural_ok for audit in selected_audits)),
            "quality_issues": float(sum(bool(audit.quality_issues) for audit in selected_audits)),
        }
    return summary


def validate_episodes(
    episodes: list[Episode],
    image_size: int | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> ValidationReport:
    audits = [audit_episode(ep, i, image_size=image_size) for i, ep in enumerate(episodes)]
    structural_failures = sum(not audit.structural_ok for audit in audits)
    gold_audits = [audit for audit in audits if audit.quality == Quality.GOLD.value]
    gold_passes = sum(audit.structural_ok and not audit.quality_issues for audit in gold_audits)
    lengths = [ep.num_transitions for ep in episodes] or [0]
    kin = [audit.kinematic_error for audit in audits] or [0.0]
    smooth = [audit.smoothness for audit in audits] or [0.0]
    stats = {
        "mean_episode_length": float(np.mean(lengths)),
        "max_kinematic_error": float(np.max(kin)),
        "mean_smoothness": float(np.mean(smooth)),
        "success_rate": float(np.mean([ep.success for ep in episodes])) if episodes else 0.0,
        "mean_image_std": float(np.mean([audit.image_std for audit in audits])) if audits else 0.0,
    }
    version_ok = schema_version == SCHEMA_VERSION
    passed = version_ok and structural_failures == 0 and gold_passes == len(gold_audits)
    return ValidationReport(
        schema_version=schema_version,
        num_episodes=len(episodes),
        num_transitions=int(sum(ep.num_transitions for ep in episodes)),
        structural_failures=structural_failures,
        gold_episodes=len(gold_audits),
        gold_standard_passes=gold_passes,
        by_quality=_quality_stats(audits, episodes),
        stats=stats,
        audits=audits,
        passed=passed,
    )


def validate_file(path: str | Path) -> ValidationReport:
    h5_path = Path(path)
    with h5py.File(h5_path, "r") as handle:
        schema_version = _as_str(handle.attrs.get("schema_version", ""))
        image_size = int(handle.attrs.get("image_size", 0))
    episodes = load_episodes(h5_path, load_images=True)
    report = validate_episodes(episodes, image_size=image_size, schema_version=schema_version)
    if schema_version != SCHEMA_VERSION:
        report.passed = False
    return report


def _as_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def format_report(report: ValidationReport) -> str:
    lines = [
        f"schema {report.schema_version}  episodes {report.num_episodes}  "
        f"transitions {report.num_transitions}",
        f"structural failures: {report.structural_failures}",
        f"gold standard: {report.gold_standard_passes}/{report.gold_episodes}",
        f"passed: {report.passed}",
        "by quality:",
    ]
    for name, stats in report.by_quality.items():
        lines.append(
            f"  {name:12s}  n={int(stats['episodes']):4d}  "
            f"success={stats['success_rate']:.2f}  "
            f"len={stats['mean_length']:.1f}  "
            f"smooth={stats['mean_smoothness']:.3f}  "
            f"issues={int(stats['quality_issues'])}"
        )
    lines.append(
        "stats: "
        + ", ".join(f"{key}={value:.4f}" for key, value in report.stats.items())
    )
    return "\n".join(lines)


def write_report(report: ValidationReport, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
