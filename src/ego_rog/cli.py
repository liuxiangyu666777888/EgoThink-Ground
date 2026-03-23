from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .egogazevqa_runner import EgoGazeVQARunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment runner for EgoGazeVQA proactive reasoning")
    parser.add_argument("--config", default="configs/egogazevqa_proactive_gaze.yaml", help="Path to YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="Inspect dataset and local frame coverage")

    sample_parser = subparsers.add_parser("sample", help="Create a balanced manifest")
    sample_parser.add_argument("--per-task", type=int, default=100, help="Number of samples per task")
    sample_parser.add_argument("--output", default="outputs/manifest_200.jsonl", help="Manifest output path")

    subparsers.add_parser("run", help="Run API inference or dry-run prompt generation")
    subparsers.add_parser("evaluate", help="Recompute summary metrics from results.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.from_file(args.config)
    runner = _build_runner(config)

    if args.command == "inspect":
        print(runner.inspect())
        return 0

    if args.command == "sample":
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = runner.sample_manifest(per_task=args.per_task, output_path=output_path)
        print({"manifest_path": str(output_path), "count": len(manifest)})
        return 0

    if args.command == "run":
        print(runner.run())
        return 0

    if args.command == "evaluate":
        print(runner.evaluate_existing())
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


def _build_runner(config: AppConfig):
    if config.experiment.dataset_kind != "egogazevqa":
        raise ValueError(f"Unsupported dataset_kind: {config.experiment.dataset_kind}")
    return EgoGazeVQARunner(config)
