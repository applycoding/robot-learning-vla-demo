"""Command line interface for data collection, training, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simreach.bc import bc_action, load_bc, save_bc, train_bc
from simreach.benchmark import BenchmarkConfig, run_benchmark
from simreach.clip_encoder import ClipEncoder
from simreach.constants import INSTRUCTIONS, Quality
from simreach.dataset import filter_quality, generate_dataset, load_episodes, stack_vla_batch, write_dataset
from simreach.env import PlanarReachEnv
from simreach.evaluate import evaluate_policy, markdown_table
from simreach.expert import expert_action
from simreach.features import proprio_features
from simreach.ppo_train import load_ppo, ppo_action, train_ppo
from simreach.validate import format_report, validate_file, write_report
from simreach.vla import (
    embedding_action,
    load_vla,
    save_vla,
    tiny_vla_action,
    train_embedding_head,
    train_tiny_vla,
)


def _quality(name: str) -> Quality:
    try:
        return Quality(name)
    except ValueError as exc:
        choices = ", ".join(item.value for item in Quality)
        raise SystemExit(f"quality must be one of: {choices}") from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="simreach",
        description="Planar-arm robot learning: demonstrations, validation, BC, PPO, and VLA.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Write an HDF5 demonstration dataset")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--gold", type=int, default=200)
    generate.add_argument("--noisy", type=int, default=80)
    generate.add_argument("--mislabeled", type=int, default=80)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--image-size", type=int, default=64)

    validate = sub.add_parser("validate", help="Check dataset schema and ground truth")
    validate.add_argument("path", type=Path)
    validate.add_argument("--report", type=Path, default=None)

    train_bc_cmd = sub.add_parser("train-bc", help="Train a behavior-cloning policy")
    train_bc_cmd.add_argument("--data", type=Path, required=True)
    train_bc_cmd.add_argument("--quality", default="gold")
    train_bc_cmd.add_argument("--out", type=Path, required=True)
    train_bc_cmd.add_argument("--epochs", type=int, default=50)
    train_bc_cmd.add_argument("--seed", type=int, default=0)

    train_ppo_cmd = sub.add_parser("train-ppo", help="Train a PPO baseline")
    train_ppo_cmd.add_argument("--timesteps", type=int, default=49152)
    train_ppo_cmd.add_argument("--out", type=Path, required=True)
    train_ppo_cmd.add_argument("--seed", type=int, default=0)

    train_vla_cmd = sub.add_parser("train-vla", help="Train a vision-language-action policy")
    train_vla_cmd.add_argument("--data", type=Path, required=True)
    train_vla_cmd.add_argument("--quality", default="gold")
    train_vla_cmd.add_argument("--encoder", choices=("tiny", "clip"), default="tiny")
    train_vla_cmd.add_argument("--out", type=Path, required=True)
    train_vla_cmd.add_argument("--epochs", type=int, default=25)
    train_vla_cmd.add_argument("--seed", type=int, default=0)

    evaluate = sub.add_parser("evaluate", help="Compare policies on shared seeds")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--bc", type=Path, default=None)
    evaluate.add_argument("--vla", type=Path, default=None)
    evaluate.add_argument("--ppo", type=Path, default=None)
    evaluate.add_argument("--expert", action="store_true")
    evaluate.add_argument("--out", type=Path, default=None)

    benchmark = sub.add_parser("benchmark", help="Run the full CPU demo and write results/")
    benchmark.add_argument("--out", type=Path, default=Path("results"))
    benchmark.add_argument("--quick", action="store_true")
    benchmark.add_argument("--with-clip", action="store_true")
    benchmark.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    command = args.command
    if command == "generate":
        episodes = generate_dataset(
            args.gold, args.noisy, args.mislabeled, seed=args.seed, image_size=args.image_size
        )
        write_dataset(args.out, episodes, args.image_size)
        print(f"wrote {len(episodes)} episodes to {args.out}")
        return
    if command == "validate":
        report = validate_file(args.path)
        print(format_report(report))
        if args.report is not None:
            write_report(report, args.report)
            print(f"wrote {args.report}")
        raise SystemExit(0 if report.passed else 1)
    if command == "train-bc":
        quality = _quality(args.quality)
        episodes = filter_quality(load_episodes(args.data, load_images=False), quality)
        checkpoint = train_bc(episodes, epochs=args.epochs, seed=args.seed)
        save_bc(checkpoint, args.out)
        print(f"saved {args.out}  final loss {checkpoint.losses[-1]:.4f}")
        return
    if command == "train-ppo":
        _model, returns = train_ppo(args.timesteps, seed=args.seed, path=args.out)
        last = float(np.mean(returns[-10:])) if returns else float("nan")
        print(f"saved {args.out}  episodes {len(returns)}  last-10 return {last:.2f}")
        return
    if command == "train-vla":
        quality = _quality(args.quality)
        episodes = filter_quality(load_episodes(args.data, load_images=True), quality)
        if args.encoder == "tiny":
            checkpoint = train_tiny_vla(episodes, epochs=args.epochs, seed=args.seed)
        else:
            checkpoint = _train_clip(episodes, args.epochs, args.seed)
        save_vla(checkpoint, args.out)
        print(f"saved {args.out}  final loss {checkpoint.losses[-1]:.4f}")
        return
    if command == "evaluate":
        summaries = _evaluate_args(args)
        text = markdown_table(summaries)
        print(text)
        if args.out is not None:
            payload = [summary.to_dict() for summary in summaries]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    if command == "benchmark":
        config = BenchmarkConfig.quick() if args.quick else BenchmarkConfig(seed=args.seed)
        if args.with_clip and not args.quick:
            config = BenchmarkConfig(seed=args.seed, with_clip=True)
        elif args.with_clip and args.quick:
            config = BenchmarkConfig.quick()
            print("ignoring --with-clip in --quick mode")
        run_benchmark(config, args.out)
        return
    raise SystemExit(f"unknown command {command}")


def _train_clip(episodes: list, epochs: int, seed: int):
    images, instruction_ids, proprio, actions = stack_vla_batch(episodes)
    texts = [INSTRUCTIONS[int(index)] for index in instruction_ids]
    encoder = ClipEncoder()
    visual, text = encoder.embed_numpy(images, texts)
    return train_embedding_head(visual, text, proprio, actions, epochs=epochs, seed=seed)


def _evaluate_args(args: argparse.Namespace):
    seeds = list(range(int(args.episodes)))
    summaries = []
    if args.expert:
        def _expert(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
            del obs
            return expert_action(env.q, env.goal)

        summaries.append(evaluate_policy("expert", "scripted", _expert, seeds, 64))
    if args.bc is not None:
        checkpoint = load_bc(args.bc)

        def _bc(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
            del env
            return bc_action(checkpoint, obs)

        summaries.append(evaluate_policy("behavior cloning", "checkpoint", _bc, seeds, 64))
    if args.vla is not None:
        checkpoint = load_vla(args.vla)
        if checkpoint.kind == "tiny":
            def _vla(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
                del obs
                return tiny_vla_action(
                    checkpoint,
                    env.render(),
                    env.instruction_id,
                    proprio_features(env.q, env.q_dot),
                )
        elif checkpoint.kind == "clip":
            encoder = ClipEncoder()

            def _vla(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
                del obs
                visual, text = encoder.embed_numpy(
                    env.render()[None, ...],
                    [INSTRUCTIONS[int(env.instruction_id)]],
                )
                return embedding_action(
                    checkpoint,
                    visual[0],
                    text[0],
                    proprio_features(env.q, env.q_dot),
                )
        else:
            raise SystemExit(f"unknown VLA kind {checkpoint.kind}")
        summaries.append(evaluate_policy("vla", checkpoint.kind, _vla, seeds, 64))
    if args.ppo is not None:
        model = load_ppo(args.ppo)

        def _ppo(env: PlanarReachEnv, obs: np.ndarray) -> np.ndarray:
            del env
            return ppo_action(model, obs)

        summaries.append(evaluate_policy("ppo", "online", _ppo, seeds, 64))
    if not summaries:
        raise SystemExit("evaluate needs at least one of --expert, --bc, --vla, --ppo")
    return summaries
