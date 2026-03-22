from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .utils import load_yaml


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    base_candidate = (base_dir / path).resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if base_candidate.exists():
        return base_candidate
    return cwd_candidate


def _resolve_path_list(base_dir: Path, values: list[str]) -> list[Path]:
    resolved = []
    for value in values:
        path = _resolve_path(base_dir, value)
        if path is not None:
            resolved.append(path)
    return resolved


@dataclass
class ExperimentConfig:
    name: str = "qwen3_vl_rog"
    dataset_kind: str = "egointention"
    output_dir: Path = Path("outputs/qwen3_vl_rog")
    dry_run: bool = False
    save_prompts: bool = True
    resume: bool = False


@dataclass
class DataConfig:
    dataset_dir: Path = Path("dataset")
    split: str = "test"
    task: str = "both"
    limit: int | None = 20
    seed: int = 42
    manifest_path: Path | None = None
    metadata_path: Path | None = None
    narration_dir: Path | None = None
    video_root: Path | None = None
    cache_dir: Path | None = None
    frame_roots: list[Path] = field(default_factory=list)
    path_replacements: dict[str, str] = field(default_factory=dict)
    skip_missing_images: bool = True
    allow_sparse_window: bool = True
    native_fps: int = 30


@dataclass
class PromptConfig:
    variant: str = "rog_thinking"
    mode: str = "sliding_window"
    window_seconds: int = 8
    sample_fps: int = 1
    max_frames: int = 8
    image_max_pixels: int = 200_704
    include_gaze_text: bool = False
    max_gaze_points: int = 12
    include_object_inventory: bool = True
    include_primary_function: bool = True
    icl_examples_per_prompt: int = 0
    silence_token: str = "<SILENCE>"
    allow_silence: bool = True
    prediction_box_format: str = "xyxy"


@dataclass
class APIConfig:
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str | None = None
    api_key_env: str = "QWEN_API_KEY"
    model: str = "Qwen3-VL-8B-Thinking"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_s: int = 90
    sample_timeout_s: int = 120
    stream: bool = False
    max_retries: int = 2

    def resolved_api_key(self) -> str | None:
        return self.api_key or os.getenv(self.api_key_env)


@dataclass
class EvaluationConfig:
    iou_thresholds: list[float] = field(default_factory=lambda: [0.3, 0.5])
    temporal_annotations_path: Path | None = None
    export_visualizations: bool = False
    visualization_limit: int = 20
    proxy_box_size_ratio: float = 0.18


@dataclass
class AppConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    api: APIConfig = field(default_factory=APIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).resolve()
        base_dir = config_path.parent
        raw = load_yaml(config_path)

        experiment = ExperimentConfig(**raw.get("experiment", {}))
        experiment.output_dir = _resolve_path(base_dir, str(experiment.output_dir)) or experiment.output_dir

        data = DataConfig(**raw.get("data", {}))
        data.dataset_dir = _resolve_path(base_dir, str(data.dataset_dir)) or data.dataset_dir
        data.manifest_path = _resolve_path(base_dir, str(data.manifest_path)) if data.manifest_path else None
        data.metadata_path = _resolve_path(base_dir, str(data.metadata_path)) if data.metadata_path else None
        data.narration_dir = _resolve_path(base_dir, str(data.narration_dir)) if data.narration_dir else None
        data.video_root = _resolve_path(base_dir, str(data.video_root)) if data.video_root else None
        data.cache_dir = _resolve_path(base_dir, str(data.cache_dir)) if data.cache_dir else None
        data.frame_roots = _resolve_path_list(base_dir, [str(item) for item in data.frame_roots])
        data.path_replacements = {
            prefix: str(_resolve_path(base_dir, target) or target)
            for prefix, target in data.path_replacements.items()
        }

        prompt = PromptConfig(**raw.get("prompt", {}))
        api = APIConfig(**raw.get("api", {}))
        evaluation = EvaluationConfig(**raw.get("evaluation", {}))
        if evaluation.temporal_annotations_path:
            evaluation.temporal_annotations_path = _resolve_path(base_dir, str(evaluation.temporal_annotations_path))

        return cls(
            experiment=experiment,
            data=data,
            prompt=prompt,
            api=api,
            evaluation=evaluation,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiment"]["output_dir"] = str(self.experiment.output_dir)
        payload["data"]["dataset_dir"] = str(self.data.dataset_dir)
        payload["data"]["manifest_path"] = str(self.data.manifest_path) if self.data.manifest_path else None
        payload["data"]["metadata_path"] = str(self.data.metadata_path) if self.data.metadata_path else None
        payload["data"]["narration_dir"] = str(self.data.narration_dir) if self.data.narration_dir else None
        payload["data"]["video_root"] = str(self.data.video_root) if self.data.video_root else None
        payload["data"]["cache_dir"] = str(self.data.cache_dir) if self.data.cache_dir else None
        payload["data"]["frame_roots"] = [str(path) for path in self.data.frame_roots]
        payload["evaluation"]["temporal_annotations_path"] = (
            str(self.evaluation.temporal_annotations_path) if self.evaluation.temporal_annotations_path else None
        )
        payload["api"]["api_key"] = "<redacted>" if self.api.api_key else None
        return payload
