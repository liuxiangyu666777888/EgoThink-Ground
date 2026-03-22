from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from .client import QwenChatClient
from .config import AppConfig
from .data import EgoIntentionDataset, build_balanced_manifest, select_examples
from .evaluation import evaluate_prediction, summarize_results
from .frames import FrameLocator
from .parsing import parse_prediction
from .prompting import PromptBuilder
from .utils import append_jsonl, dump_json, ensure_dir, load_json, read_jsonl, write_jsonl, utc_timestamp
from .visualization import draw_boxes


class ExperimentRunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset = EgoIntentionDataset(config.data.dataset_dir)
        self.assets = self.dataset.load_prompt_assets()
        self.object_inventory = sorted(set(self.dataset.inventory()) | set(self.assets.primary_function.keys()))
        self.frame_locator = FrameLocator(config.data.frame_roots, config.data.path_replacements)
        self.prompt_builder = PromptBuilder(config.prompt, self.assets, self.object_inventory)
        self.client: QwenChatClient | None = None

    def _ensure_client(self) -> QwenChatClient:
        if self.client is None:
            self.client = QwenChatClient(self.config.api)
        return self.client

    def _manifest_entries(self) -> list[dict[str, str]] | None:
        manifest_path = self.config.data.manifest_path
        if manifest_path is None or not manifest_path.exists():
            return None
        if manifest_path.suffix.lower() == ".jsonl":
            return read_jsonl(manifest_path)
        payload = load_json(manifest_path)
        if not isinstance(payload, list):
            raise ValueError(f"Manifest must be a list: {manifest_path}")
        return payload

    def load_examples(self) -> list:
        examples = self.dataset.load_examples(self.config.data.split, self.config.data.task)
        return select_examples(
            examples=examples,
            limit=self.config.data.limit,
            seed=self.config.data.seed,
            manifest_entries=self._manifest_entries(),
        )

    def sample_manifest(self, per_task: int, output_path: Path) -> list[dict[str, Any]]:
        examples = self.dataset.load_examples(self.config.data.split, task="both")
        manifest = build_balanced_manifest(examples, per_task=per_task, seed=self.config.data.seed)
        write_jsonl(output_path, manifest)
        return manifest

    def inspect(self) -> dict[str, Any]:
        examples = self.dataset.load_examples(self.config.data.split, task=self.config.data.task)
        stats: dict[str, Any] = {
            "split": self.config.data.split,
            "task": self.config.data.task,
            "count": len(examples),
            "unique_objects": len({item for example in examples for item in example.gt_objects}),
        }
        probe_examples = examples[: min(100, len(examples))]
        resolved = 0
        for example in probe_examples:
            current = self.frame_locator.resolve_frame(
                example.image_url,
                example.clip_id,
                example.frame_index,
                Path(example.image_url).suffix or ".jpeg",
            )
            if current is not None:
                resolved += 1
        stats["sampled_image_resolution_rate"] = resolved / max(1, len(probe_examples))
        stats["frame_roots"] = [str(root) for root in self.config.data.frame_roots]
        return stats

    def run(self) -> dict[str, Any]:
        output_dir = ensure_dir(self.config.experiment.output_dir)
        visuals_dir = ensure_dir(output_dir / "visuals")
        results_path = output_dir / "results.jsonl"

        dump_json(output_dir / "resolved_config.json", self.config.as_dict())
        if not self.config.experiment.resume and results_path.exists():
            results_path.unlink()

        completed_keys = set()
        if self.config.experiment.resume and results_path.exists():
            completed_keys = {
                f"{row['example']['sample_id']}:{row['example']['task']}"
                for row in read_jsonl(results_path)
            }

        if not self.config.experiment.dry_run:
            self._ensure_client()

        examples = self.load_examples()
        rows: list[dict[str, Any]] = read_jsonl(results_path) if self.config.experiment.resume and results_path.exists() else []
        visuals_written = 0

        for example in tqdm(examples, desc="Running experiment"):
            if example.key() in completed_keys:
                continue

            row = self._run_single(example, visuals_dir, visuals_written)
            rows.append(row)
            append_jsonl(results_path, row)
            if row.get("visual_written"):
                visuals_written += 1

        summary = summarize_results(rows, self.config.evaluation.iou_thresholds)
        dump_json(output_dir / "metrics.json", summary)
        return {"output_dir": str(output_dir), "results_path": str(results_path), "metrics": summary}

    def evaluate_existing(self) -> dict[str, Any]:
        results_path = self.config.experiment.output_dir / "results.jsonl"
        rows = read_jsonl(results_path)
        summary = summarize_results(rows, self.config.evaluation.iou_thresholds)
        dump_json(self.config.experiment.output_dir / "metrics.json", summary)
        return summary

    def _run_single(self, example, visuals_dir: Path, visuals_written: int) -> dict[str, Any]:
        frame_window = self.frame_locator.build_window(
            example=example,
            mode=self.config.prompt.mode,
            window_seconds=self.config.prompt.window_seconds,
            sample_fps=self.config.prompt.sample_fps,
            source_fps=self.config.data.native_fps,
            max_frames=self.config.prompt.max_frames,
            allow_sparse=self.config.data.allow_sparse_window,
        )

        row: dict[str, Any] = {
            "timestamp": utc_timestamp(),
            "status": "pending",
            "task": example.task,
            "variant": self.config.prompt.variant,
            "example": example.to_dict(),
            "frame_window": {
                "mode": frame_window.mode,
                "current_path": str(frame_window.current_path) if frame_window.current_path else None,
                "attached_frames": [str(frame.path) for frame in frame_window.frames],
                "missing_indices": frame_window.missing_indices,
            },
            "visual_written": False,
        }

        if not frame_window.frames and self.config.data.skip_missing_images:
            row["status"] = "skipped_missing_images"
            return row

        prompt_package = self.prompt_builder.build(example, frame_window)
        if self.config.experiment.save_prompts:
            row["prompt"] = {
                "system_prompt": prompt_package.system_prompt,
                "user_prompt": prompt_package.user_prompt,
                "attached_frames": prompt_package.attached_frames,
            }

        try:
            if self.config.experiment.dry_run:
                row["status"] = "dry_run"
                row["response"] = None
                row["prediction"] = None
                return row

            result = self._ensure_client().complete(prompt_package.messages)
            prediction = parse_prediction(
                result.text,
                silence_token=self.config.prompt.silence_token,
                box_format=self.config.prompt.prediction_box_format,
            )
            row.update(
                evaluate_prediction(
                    example,
                    prediction,
                    thresholds=self.config.evaluation.iou_thresholds,
                )
            )
            row["status"] = "completed"
            row["response"] = {
                "text": result.text,
                "latency_s": result.latency_s,
                "usage": result.usage,
                "finish_reason": result.finish_reason,
            }
            row["prediction"] = prediction.to_dict()

            if (
                self.config.evaluation.export_visualizations
                and visuals_written < self.config.evaluation.visualization_limit
                and frame_window.current_path is not None
                and prediction.box is not None
            ):
                caption = f"{example.sample_id}:{example.task} IoU={row['best_iou']:.3f}"
                draw_boxes(
                    image_path=frame_window.current_path,
                    gt_boxes=example.gt_boxes,
                    pred_box=prediction.box,
                    output_path=visuals_dir / f"{example.sample_id}_{example.task}.jpg",
                    caption=caption,
                )
                row["visual_written"] = True
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)

        return row
