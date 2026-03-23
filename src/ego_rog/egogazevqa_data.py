from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency fallback
    cv2 = None

from .config import RedundancyConfig

VIDEO_REF_RE = re.compile(r"(?P<dataset>ego4d|egoexo|egtea)[/\\](?P<video_id>[^/\\]+)[/\\](?P<start>\d+)_(?P<end>\d+)\.mp4$", re.IGNORECASE)
FULL_VIDEO_REF_RE = re.compile(r"(?P<dataset>ego4d|egoexo|egtea)[/\\](?P<video_id>[^/\\]+)[/\\](?P<basename>[^/\\]+)\.mp4$", re.IGNORECASE)


@dataclass
class GazePoint:
    frame: int
    x: float | None
    y: float | None
    confidence: float | None = None


@dataclass
class VideoFrame:
    path: Path
    clip_index: int
    source_frame: int


@dataclass
class EgoGazeVQASample:
    sample_id: str
    split: str
    dataset: str
    video_id: str
    file_name: str
    video_path: Path
    qa_type: str
    question: str
    answer_options: list[str]
    answer_option_map: dict[str, str]
    correct_answer: str
    start_frame: int
    end_frame: int
    gaze_sequence: list[GazePoint]

    def key(self) -> str:
        return self.sample_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "dataset": self.dataset,
            "video_id": self.video_id,
            "file_name": self.file_name,
            "video_path": str(self.video_path),
            "qa_type": self.qa_type,
            "question": self.question,
            "answer_options": self.answer_options,
            "correct_answer": self.correct_answer,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "gaze_sequence": [
                {"frame": point.frame, "x": point.x, "y": point.y, "confidence": point.confidence}
                for point in self.gaze_sequence
            ],
        }


@dataclass
class TemporalWindowAnalysis:
    original_frame_count: int
    kept_frame_count: int
    filtered_frame_count: int
    redundant_pair_count: int
    mean_optical_flow: float | None
    mean_hsv_hist_similarity: float | None
    mean_gaze_shift_norm: float | None
    redundancy_ratio: float
    low_dynamic: bool
    low_dynamic_reason: str | None
    lsfu_score: float | None
    pair_metrics: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_frame_count": self.original_frame_count,
            "kept_frame_count": self.kept_frame_count,
            "filtered_frame_count": self.filtered_frame_count,
            "redundant_pair_count": self.redundant_pair_count,
            "mean_optical_flow": self.mean_optical_flow,
            "mean_hsv_hist_similarity": self.mean_hsv_hist_similarity,
            "mean_gaze_shift_norm": self.mean_gaze_shift_norm,
            "redundancy_ratio": self.redundancy_ratio,
            "low_dynamic": self.low_dynamic,
            "low_dynamic_reason": self.low_dynamic_reason,
            "lsfu_score": self.lsfu_score,
            "pair_metrics": self.pair_metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalWindowAnalysis":
        return cls(
            original_frame_count=int(payload.get("original_frame_count", 0) or 0),
            kept_frame_count=int(payload.get("kept_frame_count", 0) or 0),
            filtered_frame_count=int(payload.get("filtered_frame_count", 0) or 0),
            redundant_pair_count=int(payload.get("redundant_pair_count", 0) or 0),
            mean_optical_flow=float(payload["mean_optical_flow"]) if payload.get("mean_optical_flow") is not None else None,
            mean_hsv_hist_similarity=(
                float(payload["mean_hsv_hist_similarity"])
                if payload.get("mean_hsv_hist_similarity") is not None
                else None
            ),
            mean_gaze_shift_norm=(
                float(payload["mean_gaze_shift_norm"]) if payload.get("mean_gaze_shift_norm") is not None else None
            ),
            redundancy_ratio=float(payload.get("redundancy_ratio", 0.0) or 0.0),
            low_dynamic=bool(payload.get("low_dynamic")),
            low_dynamic_reason=str(payload.get("low_dynamic_reason")) if payload.get("low_dynamic_reason") else None,
            lsfu_score=float(payload["lsfu_score"]) if payload.get("lsfu_score") is not None else None,
            pair_metrics=list(payload.get("pair_metrics") or []),
        )

def _normalize_split(value: str | None) -> str:
    return value.strip().lower() if value else "train"


def _normalize_answer_letter(value: str) -> str:
    value = value.strip()
    match = re.search(r"\b([A-Z])\b", value.upper())
    return match.group(1) if match else value[:1].upper()


def _parse_answer_options(raw: str | None) -> tuple[list[str], dict[str, str]]:
    if not raw:
        return [], {}
    parts = [part.strip() for part in str(raw).split("|") if part.strip()]
    option_map: dict[str, str] = {}
    normalized_parts: list[str] = []
    for idx, part in enumerate(parts):
        expected_letter = chr(ord("A") + idx)
        match = re.match(r"^\(?([A-Z])\)?[\.\:\-\s]+(.+)$", part)
        if match:
            letter = match.group(1).upper()
            text = match.group(2).strip()
        else:
            letter = expected_letter
            text = part
        option_map[letter] = text
        normalized_parts.append(f"{letter}. {text}")
    return normalized_parts, option_map


def _parse_video_ref(file_name: str) -> tuple[str, str, int, int]:
    candidate = file_name.replace("\\", "/")
    match = VIDEO_REF_RE.search(candidate)
    if match:
        return (
            match.group("dataset").lower(),
            match.group("video_id"),
            int(match.group("start")),
            int(match.group("end")),
        )

    full_match = FULL_VIDEO_REF_RE.search(candidate)
    if full_match:
        return (
            full_match.group("dataset").lower(),
            full_match.group("video_id"),
            0,
            0,
        )

    raise ValueError(f"Unsupported video reference: {file_name}")


class EgoGazeVQADataset:
    def __init__(
        self,
        dataset_dir: Path,
        metadata_path: Path | None = None,
        narration_dir: Path | None = None,
        video_root: Path | None = None,
    ):
        self.dataset_dir = dataset_dir
        self.metadata_path = metadata_path
        self.narration_dir = narration_dir or dataset_dir
        self.video_root = video_root or dataset_dir
        self._narration_cache: dict[str, dict[str, Any]] = {}

    def resolve_metadata_path(self) -> Path:
        if self.metadata_path is not None:
            return self.metadata_path
        for name in ("metadata.csv", "metadata.jsonl", "metadata.json"):
            candidate = self.dataset_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError("Could not find EgoGazeVQA metadata file.")

    def resolve_narration_path(self, dataset_name: str) -> Path:
        candidate = self.narration_dir / f"{dataset_name}.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing narration file: {candidate}")
        return candidate

    def load_narrations(self, dataset_name: str) -> dict[str, Any]:
        if dataset_name not in self._narration_cache:
            path = self.resolve_narration_path(dataset_name)
            self._narration_cache[dataset_name] = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._narration_cache[dataset_name]

    def _load_metadata_rows(self) -> list[dict[str, Any]]:
        path = self.resolve_metadata_path()
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, list):
                return payload
            raise ValueError(f"Unsupported metadata JSON format: {path}")
        raise ValueError(f"Unsupported metadata format: {path}")

    def _load_gaze_sequence(self, dataset_name: str, video_id: str, start_frame: int, end_frame: int) -> list[GazePoint]:
        narrations = self.load_narrations(dataset_name)
        video_payload = narrations.get(video_id, {})
        result: list[GazePoint] = []
        for item in video_payload.get("narrations", []):
            frame = item.get("timestamp_frame")
            if frame is None or not (start_frame <= int(frame) <= end_frame):
                continue
            gaze_info = item.get("gaze_info") or {}
            if not gaze_info:
                continue
            result.append(
                GazePoint(
                    frame=int(frame),
                    x=gaze_info.get("gaze_x"),
                    y=gaze_info.get("gaze_y"),
                    confidence=gaze_info.get("confidence"),
                )
            )
        result.sort(key=lambda point: point.frame)
        return result

    def load_examples(self, split: str = "train") -> list[EgoGazeVQASample]:
        rows = self._load_metadata_rows()
        target_split = _normalize_split(split)
        examples: list[EgoGazeVQASample] = []
        for idx, row in enumerate(rows):
            row_split = _normalize_split(row.get("split"))
            if split != "all" and row.get("split") and row_split != target_split:
                continue

            file_name = str(row.get("file_name") or row.get("video") or row.get("video_path") or "").strip()
            if not file_name:
                continue
            question = str(row.get("question") or row.get("Question") or "").strip()
            qa_type = str(row.get("qa_type") or row.get("category") or "unknown").strip().lower()
            raw_options = row.get("answer_options") or row.get("Answer Options") or ""
            correct_answer = _normalize_answer_letter(str(row.get("correct_answer") or row.get("Correct Answer") or ""))

            # The released metadata includes a small number of placeholder full-video rows
            # without any QA fields. Ignore them so inspect/run only sees valid samples.
            if not question or not str(raw_options).strip() or not correct_answer:
                continue

            dataset_name, video_id, start_frame, end_frame = _parse_video_ref(file_name)
            answer_options, answer_option_map = _parse_answer_options(str(raw_options))
            sample_id = str(row.get("sample_id") or row.get("id") or f"{dataset_name}:{video_id}:{start_frame}_{end_frame}:{idx}")
            video_path = (self.video_root / file_name).resolve()
            gaze_sequence = self._load_gaze_sequence(dataset_name, video_id, start_frame, end_frame)
            examples.append(
                EgoGazeVQASample(
                    sample_id=sample_id,
                    split=row_split,
                    dataset=dataset_name,
                    video_id=video_id,
                    file_name=file_name.replace("\\", "/"),
                    video_path=video_path,
                    qa_type=qa_type,
                    question=question,
                    answer_options=answer_options,
                    answer_option_map=answer_option_map,
                    correct_answer=correct_answer,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    gaze_sequence=gaze_sequence,
                )
            )
        return examples

    def inspect(self, split: str) -> dict[str, Any]:
        examples = self.load_examples(split)
        by_dataset: dict[str, int] = {}
        by_type: dict[str, int] = {}
        resolved = 0
        for example in examples[: min(100, len(examples))]:
            if example.video_path.exists():
                resolved += 1
        for example in examples:
            by_dataset[example.dataset] = by_dataset.get(example.dataset, 0) + 1
            by_type[example.qa_type] = by_type.get(example.qa_type, 0) + 1
        return {
            "count": len(examples),
            "split": split,
            "datasets": by_dataset,
            "qa_types": by_type,
            "sampled_video_resolution_rate": resolved / max(1, min(100, len(examples))),
        }


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _redundancy_mode_tag(config: RedundancyConfig | None) -> str:
    if config is None:
        return "red0"
    return (
        "red"
        f"_f{int(config.enable_frame_filtering)}"
        f"_of{int(round(config.optical_flow_threshold * 100))}"
        f"_hs{int(round(config.hsv_hist_similarity_threshold * 1000))}"
        f"_lf{int(round(config.low_dynamic_flow_threshold * 100))}"
        f"_lh{int(round(config.low_dynamic_hist_similarity_threshold * 1000))}"
        f"_lg{int(round(config.low_dynamic_gaze_shift_threshold * 10000))}"
        f"_mw{max(1, int(config.analysis_resize_width))}"
    )


def _resize_analysis_frame(frame_array: Any, resize_width: int) -> np.ndarray:
    rgb = np.asarray(frame_array)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = rgb.astype(np.uint8, copy=False)
    if resize_width <= 0 or rgb.shape[1] <= resize_width:
        return rgb
    scale = resize_width / float(rgb.shape[1])
    resize_height = max(1, int(round(rgb.shape[0] * scale)))
    if cv2 is not None:
        return cv2.resize(rgb, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(rgb)
    resized = image.resize((resize_width, resize_height), Image.Resampling.BILINEAR)
    return np.asarray(resized)


def _optical_flow_magnitude(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> float:
    if cv2 is None:
        prev_gray = prev_rgb.mean(axis=2).astype(np.float32)
        curr_gray = curr_rgb.mean(axis=2).astype(np.float32)
        return float(np.mean(np.abs(curr_gray - prev_gray)) / 255.0 * 10.0)

    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=128,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    if corners is None or len(corners) == 0:
        return float(np.mean(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0 * 10.0)

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
    if next_points is None or status is None:
        return 0.0
    good_prev = corners[status.flatten() == 1]
    good_next = next_points[status.flatten() == 1]
    if len(good_prev) == 0 or len(good_next) == 0:
        return 0.0
    displacements = np.linalg.norm(good_next - good_prev, axis=2)
    return float(np.median(displacements))


def _hsv_hist_similarity(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> float:
    if cv2 is None:
        prev_hist, _ = np.histogram(prev_rgb.reshape(-1, 3), bins=32, range=(0, 255), density=True)
        curr_hist, _ = np.histogram(curr_rgb.reshape(-1, 3), bins=32, range=(0, 255), density=True)
        if np.std(prev_hist) == 0 or np.std(curr_hist) == 0:
            return 1.0 if np.allclose(prev_hist, curr_hist) else 0.0
        return float(np.corrcoef(prev_hist, curr_hist)[0, 1])

    prev_hsv = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2HSV)
    curr_hsv = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2HSV)
    prev_hist = cv2.calcHist([prev_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    curr_hist = cv2.calcHist([curr_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(curr_hist, curr_hist)
    return float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL))


def _nearest_gaze_point(sample: EgoGazeVQASample, source_frame: int) -> GazePoint | None:
    if not sample.gaze_sequence:
        return None
    return min(sample.gaze_sequence, key=lambda point: abs(point.frame - source_frame))


def _gaze_shift_norm(sample: EgoGazeVQASample, prev_source_frame: int, curr_source_frame: int) -> float | None:
    prev_gaze = _nearest_gaze_point(sample, prev_source_frame)
    curr_gaze = _nearest_gaze_point(sample, curr_source_frame)
    if prev_gaze is None or curr_gaze is None:
        return None
    if prev_gaze.x is None or prev_gaze.y is None or curr_gaze.x is None or curr_gaze.y is None:
        return None
    delta = math.hypot(float(curr_gaze.x) - float(prev_gaze.x), float(curr_gaze.y) - float(prev_gaze.y))
    return float(delta / math.sqrt(2.0))


def _compute_lsfu_score(
    config: RedundancyConfig,
    mean_flow: float | None,
    mean_hist_similarity: float | None,
    mean_gaze_shift_norm: float | None,
    redundancy_ratio: float,
) -> float:
    flow_threshold = max(config.low_dynamic_flow_threshold, 1e-6)
    hist_threshold = config.low_dynamic_hist_similarity_threshold
    gaze_threshold = max(config.low_dynamic_gaze_shift_threshold, 1e-6)
    flow_component = max(0.0, (flow_threshold - float(mean_flow or 0.0)) / flow_threshold)
    hist_component = 0.0
    if mean_hist_similarity is not None:
        hist_component = max(
            0.0,
            (float(mean_hist_similarity) - hist_threshold) / max(1e-6, 1.0 - hist_threshold),
        )
    gaze_component = 0.5 if mean_gaze_shift_norm is None else max(
        0.0,
        (gaze_threshold - float(mean_gaze_shift_norm)) / gaze_threshold,
    )
    raw_score = 1.0 + 2.6 * flow_component + 2.4 * hist_component + 1.5 * gaze_component + 1.5 * redundancy_ratio
    return float(math.log1p(raw_score))


def _analyze_temporal_window(
    frame_pairs: list[tuple[int, Any]],
    sample: EgoGazeVQASample,
    config: RedundancyConfig | None,
) -> tuple[list[tuple[int, Any]], TemporalWindowAnalysis]:
    if not frame_pairs:
        summary = TemporalWindowAnalysis(
            original_frame_count=0,
            kept_frame_count=0,
            filtered_frame_count=0,
            redundant_pair_count=0,
            mean_optical_flow=None,
            mean_hsv_hist_similarity=None,
            mean_gaze_shift_norm=None,
            redundancy_ratio=0.0,
            low_dynamic=False,
            low_dynamic_reason=None,
            lsfu_score=None,
            pair_metrics=[],
        )
        return [], summary

    if config is None:
        config = RedundancyConfig()

    kept: list[tuple[int, Any]] = [frame_pairs[0]]
    pair_metrics: list[dict[str, Any]] = []
    motion_values: list[float] = []
    hist_values: list[float] = []
    gaze_values: list[float] = []
    redundant_pair_count = 0

    for idx in range(1, len(frame_pairs)):
        prev_clip_index, prev_frame = kept[-1]
        curr_clip_index, curr_frame = frame_pairs[idx]
        prev_rgb = _resize_analysis_frame(prev_frame, config.analysis_resize_width)
        curr_rgb = _resize_analysis_frame(curr_frame, config.analysis_resize_width)
        motion_score = _optical_flow_magnitude(prev_rgb, curr_rgb)
        hist_similarity = _hsv_hist_similarity(prev_rgb, curr_rgb)
        prev_source_frame = sample.start_frame + int(prev_clip_index)
        curr_source_frame = sample.start_frame + int(curr_clip_index)
        gaze_shift_norm = _gaze_shift_norm(sample, prev_source_frame, curr_source_frame)
        is_redundant = (
            motion_score < config.optical_flow_threshold
            and hist_similarity >= config.hsv_hist_similarity_threshold
        )
        pair_metrics.append(
            {
                "prev_clip_index": int(prev_clip_index),
                "curr_clip_index": int(curr_clip_index),
                "prev_source_frame": int(prev_source_frame),
                "curr_source_frame": int(curr_source_frame),
                "optical_flow": motion_score,
                "hsv_hist_similarity": hist_similarity,
                "gaze_shift_norm": gaze_shift_norm,
                "redundant": is_redundant,
            }
        )
        motion_values.append(motion_score)
        hist_values.append(hist_similarity)
        if gaze_shift_norm is not None:
            gaze_values.append(gaze_shift_norm)
        if is_redundant:
            redundant_pair_count += 1
        if not config.enable_frame_filtering or not is_redundant or idx == len(frame_pairs) - 1:
            kept.append((curr_clip_index, curr_frame))

    required_frames = min(len(frame_pairs), max(1, config.min_frames_after_filter))
    if len(kept) < required_frames:
        selected_indices = {int(item[0]) for item in kept}
        for candidate in reversed(frame_pairs):
            if int(candidate[0]) in selected_indices:
                continue
            kept.append(candidate)
            selected_indices.add(int(candidate[0]))
            if len(kept) >= required_frames:
                break
        kept.sort(key=lambda item: int(item[0]))

    original_count = len(frame_pairs)
    kept_count = len(kept)
    filtered_count = max(0, original_count - kept_count)
    redundancy_ratio = redundant_pair_count / max(1, original_count - 1) if original_count > 1 else 0.0
    mean_flow = _safe_mean(motion_values)
    mean_hist = _safe_mean(hist_values)
    mean_gaze = _safe_mean(gaze_values)
    visual_static = (
        mean_flow is not None
        and mean_hist is not None
        and mean_flow <= config.low_dynamic_flow_threshold
        and mean_hist >= config.low_dynamic_hist_similarity_threshold
    )
    gaze_static = mean_gaze is None or mean_gaze <= config.low_dynamic_gaze_shift_threshold
    redundancy_high = redundancy_ratio >= config.low_dynamic_redundancy_ratio_threshold
    low_dynamic = bool(visual_static and gaze_static and (redundancy_high or original_count <= required_frames))
    reasons: list[str] = []
    if visual_static:
        reasons.append("visual_static")
    if gaze_static:
        reasons.append("gaze_static")
    if redundancy_high:
        reasons.append("high_redundancy")
    lsfu_score = _compute_lsfu_score(config, mean_flow, mean_hist, mean_gaze, redundancy_ratio)

    summary = TemporalWindowAnalysis(
        original_frame_count=original_count,
        kept_frame_count=kept_count,
        filtered_frame_count=filtered_count,
        redundant_pair_count=redundant_pair_count,
        mean_optical_flow=mean_flow,
        mean_hsv_hist_similarity=mean_hist,
        mean_gaze_shift_norm=mean_gaze,
        redundancy_ratio=redundancy_ratio,
        low_dynamic=low_dynamic,
        low_dynamic_reason=",".join(reasons) if reasons else None,
        lsfu_score=lsfu_score,
        pair_metrics=pair_metrics,
    )
    return kept, summary


def select_egogazevqa_examples(
    examples: list[EgoGazeVQASample],
    limit: int | None,
    seed: int,
    manifest_entries: list[dict[str, str]] | None = None,
) -> list[EgoGazeVQASample]:
    if manifest_entries:
        index = {example.key(): example for example in examples}
        selected = [index[row["sample_id"]] for row in manifest_entries if row.get("sample_id") in index]
        if limit is None or limit >= len(selected):
            return selected
        return selected[:limit]
    if limit is None or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    return pool[:limit]


def build_egogazevqa_manifest(
    examples: list[EgoGazeVQASample],
    per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[EgoGazeVQASample]] = {}
    for example in examples:
        grouped.setdefault(example.qa_type, []).append(example)
    manifest: list[dict[str, Any]] = []
    for group_name, group_examples in sorted(grouped.items()):
        rng.shuffle(group_examples)
        for example in group_examples[:per_group]:
            manifest.append(
                {
                    "sample_id": example.sample_id,
                    "dataset": example.dataset,
                    "qa_type": example.qa_type,
                    "video_id": example.video_id,
                }
            )
    return manifest


def _uniform_indices(length: int, max_frames: int) -> list[int]:
    if length <= 0:
        return []
    if length <= max_frames:
        return list(range(length))
    if max_frames <= 1:
        return [0]
    step = (length - 1) / float(max_frames - 1)
    return sorted({min(length - 1, round(step * idx)) for idx in range(max_frames)})


def _temporal_window_indices(
    length: int,
    max_frames: int,
    window_seconds: int | None,
    source_fps: int,
    sample_fps: int,
    tail_only: bool,
) -> list[int]:
    if length <= 0:
        return []
    if not tail_only:
        return _uniform_indices(length, max_frames)

    if window_seconds is None or window_seconds <= 0 or source_fps <= 0:
        window_start = 0
    else:
        window_size = max(1, int(window_seconds * source_fps))
        window_start = max(0, length - window_size)

    if sample_fps > 0 and source_fps > 0:
        stride = max(1, int(round(source_fps / float(sample_fps))))
        candidates = list(range(window_start, length, stride))
        if not candidates or candidates[-1] != length - 1:
            candidates.append(length - 1)
        if len(candidates) <= max_frames:
            return sorted(set(candidates))
        selected = _uniform_indices(len(candidates), max_frames)
        return [candidates[idx] for idx in selected]

    window = list(range(window_start, length))
    if len(window) <= max_frames:
        return window
    selected = _uniform_indices(len(window), max_frames)
    return [window[idx] for idx in selected]


def decode_sampled_frames(
    sample: EgoGazeVQASample,
    max_frames: int,
    cache_dir: Path,
    window_seconds: int | None = None,
    source_fps: int = 30,
    sample_fps: int = 1,
    tail_only: bool = False,
    redundancy_config: RedundancyConfig | None = None,
) -> tuple[list[VideoFrame], TemporalWindowAnalysis]:
    if not sample.video_path.exists():
        return [], TemporalWindowAnalysis(
            original_frame_count=0,
            kept_frame_count=0,
            filtered_frame_count=0,
            redundant_pair_count=0,
            mean_optical_flow=None,
            mean_hsv_hist_similarity=None,
            mean_gaze_shift_norm=None,
            redundancy_ratio=0.0,
            low_dynamic=False,
            low_dynamic_reason=None,
            lsfu_score=None,
            pair_metrics=[],
        )

    mode_tag = f"tail_w{window_seconds or 0}_sf{sample_fps}" if tail_only else "uniform"
    mode_tag = f"{mode_tag}_{_redundancy_mode_tag(redundancy_config)}"
    sample_cache = cache_dir / f"{re.sub(r'[^a-zA-Z0-9._-]+', '_', sample.sample_id)}__{mode_tag}_f{max_frames}"
    sample_cache.mkdir(parents=True, exist_ok=True)
    summary_path = sample_cache / "window_analysis.json"

    existing = sorted(sample_cache.glob("*.jpg"))
    if existing and summary_path.exists():
        frames: list[VideoFrame] = []
        for path in existing:
            match = re.match(r"(?P<clip>\d{4})_f(?P<frame>\d+)\.jpg$", path.name)
            if not match:
                continue
            frames.append(
                VideoFrame(
                    path=path,
                    clip_index=int(match.group("clip")),
                    source_frame=int(match.group("frame")),
                )
            )
        if frames:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            return frames, TemporalWindowAnalysis.from_dict(payload)

    try:
        reader = imageio.get_reader(str(sample.video_path))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open video backend for {sample.video_path}. "
            "Install `imageio-ffmpeg` or `imageio[pyav]` to decode mp4 clips."
        ) from exc
    try:
        frame_count = reader.count_frames()
    except Exception:
        materialized = [frame for frame in reader]
        frame_count = len(materialized)
        reader.close()
        indices = _temporal_window_indices(
            frame_count,
            max_frames=max_frames,
            window_seconds=window_seconds,
            source_fps=source_fps,
            sample_fps=sample_fps,
            tail_only=tail_only,
        )
        return _write_sampled_frames(sample_cache, materialized, indices, sample, redundancy_config)

    indices = _temporal_window_indices(
        frame_count,
        max_frames=max_frames,
        window_seconds=window_seconds,
        source_fps=source_fps,
        sample_fps=sample_fps,
        tail_only=tail_only,
    )
    frames: list[tuple[int, Any]] = []
    for idx in indices:
        try:
            frames.append((idx, reader.get_data(idx)))
        except Exception:
            continue
    reader.close()
    return _write_sampled_frames(sample_cache, frames, None, sample, redundancy_config)


def _write_sampled_frames(
    sample_cache: Path,
    frames_or_materialized: list[Any],
    indices: list[int] | None,
    sample: EgoGazeVQASample,
    redundancy_config: RedundancyConfig | None,
) -> tuple[list[VideoFrame], TemporalWindowAnalysis]:
    if indices is None:
        frame_pairs = frames_or_materialized
    else:
        frame_pairs = [(idx, frames_or_materialized[idx]) for idx in indices]
    filtered_pairs, summary = _analyze_temporal_window(frame_pairs, sample, redundancy_config)
    result: list[VideoFrame] = []
    for clip_index, frame_array in filtered_pairs:
        if isinstance(frame_array, tuple):
            clip_index, frame_array = frame_array
        source_frame = sample.start_frame + int(clip_index)
        output_path = sample_cache / f"{int(clip_index):04d}_f{int(source_frame)}.jpg"
        imageio.imwrite(output_path, frame_array)
        result.append(VideoFrame(path=output_path, clip_index=int(clip_index), source_frame=int(source_frame)))
    (sample_cache / "window_analysis.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result, summary
