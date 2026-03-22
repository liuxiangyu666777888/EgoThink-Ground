from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio


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
) -> list[VideoFrame]:
    if not sample.video_path.exists():
        return []

    mode_tag = f"tail_w{window_seconds or 0}_sf{sample_fps}" if tail_only else "uniform"
    sample_cache = cache_dir / f"{re.sub(r'[^a-zA-Z0-9._-]+', '_', sample.sample_id)}__{mode_tag}_f{max_frames}"
    sample_cache.mkdir(parents=True, exist_ok=True)

    existing = sorted(sample_cache.glob("*.jpg"))
    if existing:
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
            return frames

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
        return _write_sampled_frames(sample_cache, materialized, indices, sample)

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
    return _write_sampled_frames(sample_cache, frames, None, sample)


def _write_sampled_frames(
    sample_cache: Path,
    frames_or_materialized: list[Any],
    indices: list[int] | None,
    sample: EgoGazeVQASample,
) -> list[VideoFrame]:
    if indices is None:
        frame_pairs = frames_or_materialized
    else:
        frame_pairs = [(idx, frames_or_materialized[idx]) for idx in indices]
    result: list[VideoFrame] = []
    for clip_index, frame_array in frame_pairs:
        if isinstance(frame_array, tuple):
            clip_index, frame_array = frame_array
        source_frame = sample.start_frame + int(clip_index)
        output_path = sample_cache / f"{int(clip_index):04d}_f{int(source_frame)}.jpg"
        imageio.imwrite(output_path, frame_array)
        result.append(VideoFrame(path=output_path, clip_index=int(clip_index), source_frame=int(source_frame)))
    return result
