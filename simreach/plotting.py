"""Figures and GIFs written by the benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from simreach.evaluate import EvalSummary

_COLORS = {
    "expert": "#0f766e",
    "behavior cloning": "#2563eb",
    "vla": "#7c3aed",
    "vla-clip": "#6d28d9",
    "ppo": "#d97706",
    "random": "#64748b",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )


def save_gif(frames: list[np.ndarray], path: str | Path, duration_ms: int = 70) -> None:
    if not frames:
        raise ValueError("no frames to write")
    images = [Image.fromarray(np.ascontiguousarray(frame)) for frame in frames]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def plot_success_rates(summaries: list[EvalSummary], path: str | Path) -> None:
    _style()
    labels = [f"{summary.method}\n{summary.data}" for summary in summaries]
    values = [100.0 * summary.success_rate for summary in summaries]
    colors = [_COLORS.get(summary.method, "#334155") for summary in summaries]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    positions = np.arange(len(summaries))
    bars = ax.bar(positions, values, color=colors, width=0.72)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success (%)")
    ax.set_title("Evaluation success on held-out seeds")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2.0, 96),
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_data_quality(summaries: list[EvalSummary], path: str | Path) -> None:
    """Grouped bars of imitation success by demonstration tier."""
    _style()
    tiers = ["gold", "noisy", "mislabeled"]
    methods = ["behavior cloning", "vla"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.36
    positions = np.arange(len(tiers))
    for offset, method in enumerate(methods):
        values = []
        for tier in tiers:
            match = [
                summary.success_rate
                for summary in summaries
                if summary.method == method and summary.data == tier
            ]
            values.append(100.0 * match[0] if match else 0.0)
        ax.bar(
            positions + (offset - 0.5) * width,
            values,
            width=width,
            color=_COLORS[method],
            label=method,
        )
    ax.set_xticks(positions, tiers)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success (%)")
    ax.set_title("How demonstration quality changes imitation")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_curves(
    bc_losses: dict[str, list[float]],
    vla_losses: dict[str, list[float]],
    ppo_returns: list[float],
    path: str | Path,
) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for name, losses in bc_losses.items():
        axes[0].plot(np.arange(1, len(losses) + 1), losses, label=name)
    axes[0].set_title("Behavior cloning loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend(frameon=False, fontsize=9)

    for name, losses in vla_losses.items():
        axes[1].plot(np.arange(1, len(losses) + 1), losses, label=name)
    axes[1].set_title("VLA loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend(frameon=False, fontsize=9)

    if ppo_returns:
        window = max(1, len(ppo_returns) // 20)
        kernel = np.ones(window) / window
        smoothed = np.convolve(np.asarray(ppo_returns, dtype=np.float64), kernel, mode="valid")
        axes[2].plot(smoothed, color=_COLORS["ppo"])
    axes[2].set_title("PPO episode return")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Return")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
