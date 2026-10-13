"""End-to-end CPU benchmark: data, validation, BC, PPO, VLA, and plots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from simreach.bc import bc_action, save_bc, train_bc
from simreach.clip_encoder import DEFAULT_CLIP_NAME, ClipEncoder
from simreach.constants import IMAGE_SIZE, INSTRUCTIONS, Quality
from simreach.dataset import (
    filter_quality,
    generate_dataset,
    stack_vla_batch,
    write_dataset,
)
from simreach.env import PlanarReachEnv
from simreach.evaluate import (
    EvalSummary,
    evaluate_policy,
    markdown_table,
    run_episode,
)
from simreach.expert import expert_action
from simreach.features import proprio_features
from simreach.plotting import plot_curves, plot_data_quality, plot_success_rates, save_gif
from simreach.ppo_train import ppo_action, train_ppo
from simreach.validate import format_report, validate_episodes, write_report
from simreach.vla import (
    embedding_action,
    save_vla,
    tiny_vla_action,
    train_embedding_head,
    train_tiny_vla,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 0
    image_size: int = IMAGE_SIZE
    n_gold: int = 200
    n_noisy: int = 80
    n_mislabeled: int = 80
    bc_epochs: int = 50
    bc_hidden: int = 64
    bc_weight_decay: float = 1e-3
    bc_obs_noise: float = 0.05
    vla_epochs: int = 25
    ppo_timesteps: int = 49152
    eval_episodes: int = 40
    with_clip: bool = False
    clip_model: str = DEFAULT_CLIP_NAME

    @staticmethod
    def quick() -> BenchmarkConfig:
        return BenchmarkConfig(
            n_gold=3,
            n_noisy=2,
            n_mislabeled=2,
            bc_epochs=2,
            bc_weight_decay=0.0,
            bc_obs_noise=0.0,
            vla_epochs=1,
            ppo_timesteps=512,
            eval_episodes=2,
            with_clip=False,
        )


def run_benchmark(config: BenchmarkConfig, out_dir: str | Path) -> dict[str, object]:
    """Generate data, train every baseline, evaluate on shared seeds, write artifacts."""
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    print(
        f"generating demos gold={config.n_gold} noisy={config.n_noisy} "
        f"mislabeled={config.n_mislabeled}"
    )
    episodes = generate_dataset(
        config.n_gold,
        config.n_noisy,
        config.n_mislabeled,
        seed=config.seed,
        image_size=config.image_size,
    )
    dataset_path = out / "dataset" / "reach_demos.h5"
    write_dataset(dataset_path, episodes, config.image_size)
    report = validate_episodes(episodes, image_size=config.image_size)
    write_report(report, out / "validation.json")
    print(format_report(report))

    seeds = list(range(config.eval_episodes))
    summaries: list[EvalSummary] = []

    def _expert(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
        del obs
        return expert_action(env.q, env.goal)

    def _random(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
        del env, obs
        return np.random.default_rng(config.seed + 17).uniform(-1.0, 1.0, size=2).astype(np.float32)

    print("evaluating expert and random")
    summaries.append(
        evaluate_policy("expert", "scripted", _expert, seeds, config.image_size)
    )
    random_rng = np.random.default_rng(config.seed + 17)

    def _random_varying(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
        del env, obs
        return random_rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    summaries.append(
        evaluate_policy("random", "none", _random_varying, seeds, config.image_size)
    )

    bc_losses: dict[str, list[float]] = {}
    bc_checkpoints = {}
    for quality in Quality:
        subset = filter_quality(episodes, quality)
        print(f"training behavior cloning on {quality.value} ({len(subset)} episodes)")
        checkpoint = train_bc(
            subset,
            epochs=config.bc_epochs,
            seed=config.seed,
            hidden=config.bc_hidden,
            weight_decay=config.bc_weight_decay,
            obs_noise=config.bc_obs_noise,
        )
        save_bc(checkpoint, checkpoints / f"bc_{quality.value}.pt")
        bc_losses[quality.value] = checkpoint.losses
        bc_checkpoints[quality.value] = checkpoint

        def _act(env: PlanarReachEnv, obs: np.ndarray, held: object = checkpoint) -> np.ndarray:
            del env
            return bc_action(held, obs)  # type: ignore[arg-type]

        summaries.append(
            evaluate_policy(
                "behavior cloning",
                quality.value,
                _act,
                seeds,
                config.image_size,
            )
        )

    vla_losses: dict[str, list[float]] = {}
    vla_checkpoints = {}
    for quality in Quality:
        subset = filter_quality(episodes, quality)
        print(f"training tiny VLA on {quality.value} ({len(subset)} episodes)")
        checkpoint = train_tiny_vla(
            subset,
            epochs=config.vla_epochs,
            seed=config.seed,
        )
        save_vla(checkpoint, checkpoints / f"vla_{quality.value}.pt")
        vla_losses[quality.value] = checkpoint.losses
        vla_checkpoints[quality.value] = checkpoint

        def _act_vla(env: PlanarReachEnv, obs: np.ndarray, held: object = checkpoint) -> np.ndarray:
            del obs
            return tiny_vla_action(
                held,  # type: ignore[arg-type]
                env.render(),
                env.instruction_id,
                proprio_features(env.q, env.q_dot),
            )

        summaries.append(
            evaluate_policy("vla", quality.value, _act_vla, seeds, config.image_size)
        )

    print(f"training PPO for {config.ppo_timesteps} steps")
    ppo_model, ppo_returns = train_ppo(
        config.ppo_timesteps,
        seed=config.seed,
        path=checkpoints / "ppo",
    )

    def _act_ppo(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
        del env
        return ppo_action(ppo_model, obs)

    summaries.append(
        evaluate_policy("ppo", "online", _act_ppo, seeds, config.image_size)
    )

    if config.with_clip:
        print(f"encoding gold demos with {config.clip_model}")
        gold = filter_quality(episodes, Quality.GOLD)
        images, instruction_ids, proprio, actions = stack_vla_batch(gold)
        texts = [INSTRUCTIONS[int(index)] for index in instruction_ids]
        encoder = ClipEncoder(model_name=config.clip_model)
        visual, text = encoder.embed_numpy(images, texts)
        clip_checkpoint = train_embedding_head(
            visual,
            text,
            proprio,
            actions,
            epochs=max(config.vla_epochs, 1),
            seed=config.seed,
        )
        save_vla(clip_checkpoint, checkpoints / "vla_clip_gold.pt")
        vla_losses["clip-gold"] = clip_checkpoint.losses

        def _act_clip(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
            del obs
            frame = env.render()[None, ...]
            visual_row, text_row = encoder.embed_numpy(
                frame, [INSTRUCTIONS[int(env.instruction_id)]]
            )
            return embedding_action(
                clip_checkpoint,
                visual_row[0],
                text_row[0],
                proprio_features(env.q, env.q_dot),
            )

        summaries.append(
            evaluate_policy("vla-clip", "gold", _act_clip, seeds, config.image_size)
        )

    hero = [
        summary
        for summary in summaries
        if (summary.method, summary.data)
        in {
            ("expert", "scripted"),
            ("behavior cloning", "gold"),
            ("vla", "gold"),
            ("vla-clip", "gold"),
            ("ppo", "online"),
            ("random", "none"),
        }
    ]
    plot_success_rates(hero, out / "success_rates.png")
    plot_data_quality(summaries, out / "data_quality.png")
    plot_curves(bc_losses, vla_losses, ppo_returns, out / "learning_curves.png")
    _write_gif(out / "expert.gif", _expert, seeds, summaries[0])
    gold_bc = next(
        summary
        for summary in summaries
        if summary.method == "behavior cloning" and summary.data == "gold"
    )
    _write_gif(
        out / "bc.gif",
        lambda env, obs, held=bc_checkpoints["gold"]: bc_action(held, obs),
        seeds,
        gold_bc,
    )
    table = markdown_table(summaries)
    (out / "table.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    metrics: dict[str, object] = {
        "config": asdict(config),
        "validation_passed": report.passed,
        "validation_summary": {
            "structural_failures": report.structural_failures,
            "gold_standard_passes": report.gold_standard_passes,
            "gold_episodes": report.gold_episodes,
            "by_quality": report.by_quality,
            "stats": report.stats,
        },
        "results": [summary.to_dict() for summary in summaries],
        "bc_final_loss": {name: values[-1] if values else None for name, values in bc_losses.items()},
        "vla_final_loss": {name: values[-1] if values else None for name, values in vla_losses.items()},
        "ppo_episodes": len(ppo_returns),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _write_gif(
    path: Path,
    act: object,
    seeds: list[int],
    summary: EvalSummary,
) -> None:
    chosen = seeds[0]
    for seed, success in zip(summary.seeds, summary.episode_success):
        if success:
            chosen = seed
            break
    env = PlanarReachEnv(image_size=IMAGE_SIZE)
    result = run_episode(env, act, chosen, record_size=240)  # type: ignore[arg-type]
    save_gif(result.frames, path)
