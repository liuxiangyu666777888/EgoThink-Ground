from __future__ import annotations

import base64
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from requests import HTTPError, RequestException
from tqdm import tqdm

from .client import QwenChatClient
from .config import AppConfig
from .egogazevqa_data import (
    EgoGazeVQADataset,
    EgoGazeVQASample,
    VideoFrame,
    build_egogazevqa_manifest,
    decode_sampled_frames,
    select_egogazevqa_examples,
)
from .geometry import Box
from .parsing import Prediction, parse_prediction
from .utils import append_jsonl, dump_json, ensure_dir, load_json, read_jsonl, write_jsonl, utc_timestamp
from .visualization import draw_boxes


ANSWER_PATTERNS = [
    re.compile(r"^\s*([A-Z])\s*$", re.IGNORECASE),
    re.compile(r"\banswer\s*[:=]?\s*([A-Z])\b", re.IGNORECASE),
    re.compile(r"\boption\s*([A-Z])\b", re.IGNORECASE),
    re.compile(r"\bchoose\s*([A-Z])\b", re.IGNORECASE),
    re.compile(r"\(([A-Z])\)"),
]

INTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "under",
    "was",
    "were",
    "while",
    "with",
}


@dataclass
class VQAPrediction:
    raw_text: str
    answer: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"raw_text": self.raw_text, "answer": self.answer}


@dataclass
class GazeAnchor:
    source_frame: int
    x_norm: float
    y_norm: float
    confidence: float | None
    width: int
    height: int
    x_px: float
    y_px: float
    proxy_box: Box
    frame_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "x_norm": self.x_norm,
            "y_norm": self.y_norm,
            "confidence": self.confidence,
            "width": self.width,
            "height": self.height,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "proxy_box_xyxy": self.proxy_box.as_list(),
            "frame_path": str(self.frame_path),
        }


def _image_to_data_url(path: Path, max_pixels: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_pixels and image.width * image.height > max_pixels:
            scale = math.sqrt(max_pixels / float(image.width * image.height))
            resized = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
            image = image.resize(resized)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _frame_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.width, image.height


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _stem_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _content_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    tokens = []
    for token in _normalize_text(value).split():
        stemmed = _stem_token(token)
        if len(stemmed) <= 1 or stemmed in INTENT_STOPWORDS:
            continue
        tokens.append(stemmed)
    return set(tokens)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_filename(value: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', value).strip('._') or "sample"


def _token_overlap_stats(predicted_tokens: set[str], reference_tokens: set[str]) -> tuple[int, float, float, float]:
    if not predicted_tokens or not reference_tokens:
        return 0, 0.0, 0.0, 0.0
    overlap = len(predicted_tokens & reference_tokens)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    shorter_coverage = overlap / min(len(predicted_tokens), len(reference_tokens))
    return overlap, precision, recall, shorter_coverage


def _canonicalize_box(box: Box) -> Box:
    return Box(
        xmin=min(box.xmin, box.xmax),
        ymin=min(box.ymin, box.ymax),
        xmax=max(box.xmin, box.xmax),
        ymax=max(box.ymin, box.ymax),
    )


def _coerce_box_to_frame(box: Box | None, width: int, height: int) -> tuple[Box | None, str | None]:
    if box is None:
        return None, None
    normalized_box = _canonicalize_box(box)
    if width <= 0 or height <= 0:
        return normalized_box, "unknown"

    values = normalized_box.as_list()
    if all(-0.05 <= value <= 1.05 for value in values):
        scaled = Box(
            xmin=normalized_box.xmin * width,
            ymin=normalized_box.ymin * height,
            xmax=normalized_box.xmax * width,
            ymax=normalized_box.ymax * height,
        )
        return scaled.clip(width, height), "normalized_0_1"

    exceeds_frame = (
        normalized_box.xmax > width * 1.05
        or normalized_box.ymax > height * 1.05
        or normalized_box.xmin < -width * 0.05
        or normalized_box.ymin < -height * 0.05
    )
    if exceeds_frame and all(-5.0 <= value <= 1005.0 for value in values):
        scaled = Box(
            xmin=(normalized_box.xmin / 1000.0) * width,
            ymin=(normalized_box.ymin / 1000.0) * height,
            xmax=(normalized_box.xmax / 1000.0) * width,
            ymax=(normalized_box.ymax / 1000.0) * height,
        )
        return scaled.clip(width, height), "normalized_0_1000"

    return normalized_box.clip(width, height), "pixel"


def _refine_box_with_gaze(
    box: Box | None,
    gaze_anchor: GazeAnchor | None,
    width: int,
    height: int,
) -> tuple[Box | None, str | None]:
    if box is None or gaze_anchor is None or width <= 0 or height <= 0:
        return box, None
    if box.contains(gaze_anchor.x_px, gaze_anchor.y_px):
        return box, None

    min_width = max(width * 0.08, 4.0)
    min_height = max(height * 0.08, 4.0)
    refined = Box.from_center(
        gaze_anchor.x_px,
        gaze_anchor.y_px,
        max(box.width, min_width),
        max(box.height, min_height),
    ).clip(width, height)
    return refined, "gaze_anchor_recenter"


def _extract_row_gaze_anchor(row: dict[str, Any]) -> dict[str, Any] | None:
    gaze_anchor = row.get("gaze_anchor")
    if isinstance(gaze_anchor, dict) and gaze_anchor:
        return gaze_anchor
    prompt = row.get("prompt")
    if isinstance(prompt, dict):
        prompt_anchor = prompt.get("gaze_anchor")
        if isinstance(prompt_anchor, dict) and prompt_anchor:
            return prompt_anchor
    return None


def _intent_similarity(predicted: str | None, reference: str | None) -> float | None:
    if not predicted or not reference:
        return None

    predicted_norm = _normalize_text(predicted)
    reference_norm = _normalize_text(reference)
    if not predicted_norm or not reference_norm:
        return None

    sequence_ratio = SequenceMatcher(None, predicted_norm, reference_norm).ratio()
    predicted_tokens = _content_tokens(predicted_norm)
    reference_tokens = _content_tokens(reference_norm)
    if not predicted_tokens or not reference_tokens:
        return sequence_ratio

    overlap, precision, recall, _ = _token_overlap_stats(predicted_tokens, reference_tokens)
    if overlap == 0:
        return sequence_ratio
    token_f1 = 2 * precision * recall / (precision + recall)
    return max(sequence_ratio, token_f1)


def _intent_match(predicted: str | None, reference: str | None) -> bool:
    similarity = _intent_similarity(predicted, reference)
    if similarity is None:
        return False

    predicted_norm = _normalize_text(predicted or "")
    reference_norm = _normalize_text(reference or "")
    if predicted_norm and reference_norm:
        if reference_norm in predicted_norm or predicted_norm in reference_norm:
            return True

    predicted_tokens = _content_tokens(predicted_norm)
    reference_tokens = _content_tokens(reference_norm)
    overlap, precision, recall, shorter_coverage = _token_overlap_stats(predicted_tokens, reference_tokens)
    if overlap >= 4 and shorter_coverage >= 0.7:
        return True
    if overlap >= 4 and similarity >= 0.62:
        return True
    if overlap >= 5 and precision >= 0.65 and recall >= 0.5:
        return True
    return similarity >= 0.68


def _derive_non_thinking_model(model_name: str) -> str | None:
    fallback = re.sub(r"(?i)(?:[-_ ]?thinking)\b", "", model_name).strip("-_ ")
    if not fallback or fallback.lower() == model_name.lower():
        return None
    return fallback


def _is_retryable_inference_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        response = exc.response
        return response is not None and response.status_code >= 500
    return isinstance(exc, RequestException)


def _remaining_timeout_s(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.perf_counter())


def _row_intent_match(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction")
    if isinstance(prediction, dict):
        predicted_intention = prediction.get("intention")
        reference_intention = row.get("reference_intention")
        if predicted_intention or reference_intention:
            return _intent_match(predicted_intention, reference_intention)
    if row.get("strict_intent_match") is not None:
        return bool(row.get("strict_intent_match"))
    if row.get("intent_match") is not None:
        return bool(row.get("intent_match"))
    return False


def _row_intent_consistent(row: dict[str, Any]) -> bool:
    prediction = row.get("prediction")
    predicted_intention = prediction.get("intention") if isinstance(prediction, dict) else None
    if not predicted_intention:
        if row.get("intent_consistent") is not None:
            return bool(row.get("intent_consistent"))
        return False

    predicted_answer = row.get("predicted_answer")
    reference_answer = row.get("reference_answer")
    if predicted_answer and reference_answer and predicted_answer == reference_answer:
        return True

    if row.get("reference_intention"):
        return _intent_match(predicted_intention, row.get("reference_intention"))
    if row.get("intent_consistent") is not None:
        return bool(row.get("intent_consistent"))
    return False


def _effective_prediction_box(row: dict[str, Any]) -> Box | None:
    prediction = row.get("prediction")
    if not isinstance(prediction, dict):
        return None
    box_xyxy = prediction.get("box_xyxy")
    if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
        return None
    try:
        parsed_box = Box.from_sequence(box_xyxy, fmt="xyxy")
    except ValueError:
        return None

    gaze_anchor = _extract_row_gaze_anchor(row)
    if prediction.get("box_xyxy_raw") is not None:
        if gaze_anchor is None:
            return _canonicalize_box(parsed_box)
        width = int(gaze_anchor.get("width", 0) or 0)
        height = int(gaze_anchor.get("height", 0) or 0)
        return _canonicalize_box(parsed_box).clip(width, height)

    if gaze_anchor is None:
        return _canonicalize_box(parsed_box)
    width = int(gaze_anchor.get("width", 0) or 0)
    height = int(gaze_anchor.get("height", 0) or 0)
    effective_box, _ = _coerce_box_to_frame(parsed_box, width, height)
    return effective_box


def _model_prediction_box(row: dict[str, Any]) -> Box | None:
    prediction = row.get("prediction")
    if not isinstance(prediction, dict):
        return None
    box_xyxy = prediction.get("box_xyxy_model")
    if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
        box_xyxy = prediction.get("box_xyxy")
    if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
        return None
    try:
        parsed_box = Box.from_sequence(box_xyxy, fmt="xyxy")
    except ValueError:
        return None

    gaze_anchor = _extract_row_gaze_anchor(row)
    if gaze_anchor is None:
        return _canonicalize_box(parsed_box)
    width = int(gaze_anchor.get("width", 0) or 0)
    height = int(gaze_anchor.get("height", 0) or 0)
    return _canonicalize_box(parsed_box).clip(width, height)


def _box_alignment_metrics(box: Box | None, gaze_anchor: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "has_box": box is not None,
        "proxy_iou": None,
        "point_hit": None,
        "center_distance_norm": None,
    }
    if gaze_anchor is None or box is None:
        return metrics

    proxy_box_xyxy = gaze_anchor.get("proxy_box_xyxy")
    if not isinstance(proxy_box_xyxy, list) or len(proxy_box_xyxy) != 4:
        return metrics

    proxy_box = Box.from_sequence(proxy_box_xyxy, fmt="xyxy")
    x_px = float(gaze_anchor.get("x_px", 0.0))
    y_px = float(gaze_anchor.get("y_px", 0.0))
    width = float(gaze_anchor.get("width", 0.0))
    height = float(gaze_anchor.get("height", 0.0))
    center_x, center_y = box.center
    diagonal = math.hypot(width, height)
    metrics["proxy_iou"] = box.iou(proxy_box)
    metrics["point_hit"] = box.contains(x_px, y_px)
    metrics["center_distance_norm"] = (
        math.hypot(center_x - x_px, center_y - y_px) / diagonal if diagonal > 0 else None
    )
    return metrics


def _row_proactive_metrics(row: dict[str, Any]) -> dict[str, Any]:
    gaze_anchor = _extract_row_gaze_anchor(row)
    effective_box = _effective_prediction_box(row)
    model_box = _model_prediction_box(row)
    refined_metrics = _box_alignment_metrics(effective_box, gaze_anchor)
    raw_metrics = _box_alignment_metrics(model_box, gaze_anchor)
    metrics: dict[str, Any] = {
        "gaze_anchor_present": gaze_anchor is not None,
        "intent_consistent": _row_intent_consistent(row),
        "intent_match": _row_intent_match(row),
        "has_box": refined_metrics["has_box"],
        "proxy_iou": refined_metrics["proxy_iou"],
        "point_hit": refined_metrics["point_hit"],
        "center_distance_norm": refined_metrics["center_distance_norm"],
        "raw_has_box": raw_metrics["has_box"],
        "raw_proxy_iou": raw_metrics["proxy_iou"],
        "raw_point_hit": raw_metrics["point_hit"],
        "raw_center_distance_norm": raw_metrics["center_distance_norm"],
        "refinement_applied": bool((row.get("prediction") or {}).get("box_refinement")),
    }
    return metrics


def extract_answer_letter(text: str, option_map: dict[str, str]) -> str | None:
    stripped = text.strip()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(stripped)
        if match and match.group(1).upper() in option_map:
            return match.group(1).upper()

    tokens = re.findall(r"\b[A-Z]\b", stripped.upper())
    for token in tokens:
        if token in option_map:
            return token

    normalized_output = _normalize_text(stripped)
    for letter, option_text in option_map.items():
        candidate = _normalize_text(option_text)
        if candidate and candidate in normalized_output:
            return letter
    return None


def summarize_vqa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(group_rows)
        if total == 0:
            return {"count": 0}
        completed_rows = [row for row in group_rows if row.get("status") == "completed"]
        response_rows = [row for row in group_rows if row.get("response")]
        return {
            "count": total,
            "accuracy": sum(1 for row in group_rows if row.get("correct")) / total,
            "completed_rate": len(completed_rows) / total,
            "parsed_answer_rate": sum(1 for row in group_rows if row.get("predicted_answer")) / total,
            "mean_latency_s": (
                sum(float(row.get("response", {}).get("latency_s", 0.0)) for row in response_rows) / max(1, len(response_rows))
            ),
        }

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row.get("dataset", "unknown")].append(row)
        by_type[row.get("qa_type", "unknown")].append(row)
        by_status[row.get("status", "unknown")].append(row)
    return {
        "overall": aggregate(rows),
        "by_dataset": {name: aggregate(items) for name, items in sorted(by_dataset.items())},
        "by_qa_type": {name: aggregate(items) for name, items in sorted(by_type.items())},
        "by_status": {name: aggregate(items) for name, items in sorted(by_status.items())},
    }


def summarize_proactive(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    def aggregate(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(group_rows)
        if total == 0:
            return {"count": 0}
        completed_rows = [row for row in group_rows if row.get("status") == "completed"]
        response_rows = [row for row in group_rows if row.get("response")]
        proactive_metrics = [_row_proactive_metrics(row) for row in group_rows]
        proxy_iou_values = [
            float(metric["proxy_iou"]) for metric in proactive_metrics if metric.get("proxy_iou") is not None
        ]
        raw_proxy_iou_values = [
            float(metric["raw_proxy_iou"]) for metric in proactive_metrics if metric.get("raw_proxy_iou") is not None
        ]
        center_distance_values = [
            float(metric["center_distance_norm"])
            for metric in proactive_metrics
            if metric.get("center_distance_norm") is not None
        ]
        raw_center_distance_values = [
            float(metric["raw_center_distance_norm"])
            for metric in proactive_metrics
            if metric.get("raw_center_distance_norm") is not None
        ]

        summary: dict[str, Any] = {
            "count": total,
            "accuracy": sum(1 for row in group_rows if row.get("correct")) / total,
            "intent_accuracy": sum(1 for metric in proactive_metrics if metric.get("intent_consistent")) / total,
            "intent_consistency": sum(1 for metric in proactive_metrics if metric.get("intent_consistent")) / total,
            "strict_intent_accuracy": sum(1 for metric in proactive_metrics if metric.get("intent_match")) / total,
            "completed_rate": len(completed_rows) / total,
            "parsed_answer_rate": sum(1 for row in group_rows if row.get("predicted_answer")) / total,
            "has_box_rate": sum(1 for metric in proactive_metrics if metric.get("has_box")) / total,
            "raw_has_box_rate": sum(1 for metric in proactive_metrics if metric.get("raw_has_box")) / total,
            "silence_rate": sum(1 for row in group_rows if row.get("is_silence")) / total,
            "gaze_anchor_rate": sum(1 for metric in proactive_metrics if metric.get("gaze_anchor_present")) / total,
            "point_hit_rate": sum(1 for metric in proactive_metrics if metric.get("point_hit")) / total,
            "raw_point_hit_rate": sum(1 for metric in proactive_metrics if metric.get("raw_point_hit")) / total,
            "mean_proxy_iou": _safe_mean(proxy_iou_values),
            "raw_mean_proxy_iou": _safe_mean(raw_proxy_iou_values),
            "mean_center_distance_norm": _safe_mean(center_distance_values),
            "raw_mean_center_distance_norm": _safe_mean(raw_center_distance_values),
            "refinement_applied_rate": sum(1 for metric in proactive_metrics if metric.get("refinement_applied")) / total,
            "mean_latency_s": (
                sum(float(row.get("response", {}).get("latency_s", 0.0)) for row in response_rows) / max(1, len(response_rows))
            ),
        }
        for threshold in thresholds:
            key = f"proxy_iou_ge_{str(threshold).replace('.', '_')}"
            raw_key = f"raw_proxy_iou_ge_{str(threshold).replace('.', '_')}"
            summary[key] = sum(
                1
                for metric in proactive_metrics
                if metric.get("proxy_iou") is not None and float(metric["proxy_iou"]) >= threshold
            ) / total
            summary[raw_key] = sum(
                1
                for metric in proactive_metrics
                if metric.get("raw_proxy_iou") is not None and float(metric["raw_proxy_iou"]) >= threshold
            ) / total
        return summary

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row.get("dataset", "unknown")].append(row)
        by_type[row.get("qa_type", "unknown")].append(row)
        by_status[row.get("status", "unknown")].append(row)
    return {
        "overall": aggregate(rows),
        "by_dataset": {name: aggregate(items) for name, items in sorted(by_dataset.items())},
        "by_qa_type": {name: aggregate(items) for name, items in sorted(by_type.items())},
        "by_status": {name: aggregate(items) for name, items in sorted(by_status.items())},
    }


class EgoGazeVQARunner:
    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset = EgoGazeVQADataset(
            dataset_dir=config.data.dataset_dir,
            metadata_path=config.data.metadata_path,
            narration_dir=config.data.narration_dir,
            video_root=config.data.video_root,
        )
        self.client: QwenChatClient | None = None

    def _is_proactive_mode(self) -> bool:
        return self.config.prompt.mode == "proactive_window" or "proactive" in self.config.prompt.variant

    def _is_fast_mode(self) -> bool:
        return "fast" in self.config.prompt.variant.lower()

    def _ensure_client(self) -> QwenChatClient:
        if self.client is None:
            self.client = QwenChatClient(self.config.api)
        return self.client

    def _complete_with_recovery(
        self,
        messages: list[dict[str, Any]],
        deadline: float | None = None,
    ) -> tuple[Any, list[dict[str, Any]], float]:
        attempts: list[dict[str, Any]] = []
        started_at = time.perf_counter()

        def invoke(strategy: str, model_override: str | None, allow_failure: bool) -> Any | None:
            model_name = model_override or self.config.api.model
            remaining_timeout = _remaining_timeout_s(deadline)
            if remaining_timeout is not None and remaining_timeout <= 0:
                timeout_exc = TimeoutError("Sample-level timeout exceeded before sending request.")
                attempts.append(
                    {
                        "strategy": strategy,
                        "model": model_name,
                        "status": "error",
                        "error": str(timeout_exc),
                    }
                )
                if allow_failure:
                    return None
                raise timeout_exc
            try:
                result = self._ensure_client().complete(
                    messages,
                    model_override=model_override,
                    timeout_override=remaining_timeout,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "strategy": strategy,
                        "model": model_name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                if allow_failure:
                    return None
                raise

            attempts.append(
                {
                    "strategy": strategy,
                    "model": result.model,
                    "status": "completed",
                    "latency_s": result.latency_s,
                    "finish_reason": result.finish_reason,
                    "text_length": len(result.text.strip()),
                    "empty_text": not bool(result.text.strip()),
                }
            )
            return result

        primary = invoke(strategy="primary", model_override=None, allow_failure=False)
        assert primary is not None
        best_result = primary
        if primary.text.strip():
            return primary, attempts, time.perf_counter() - started_at

        retry = invoke(strategy="empty_text_retry", model_override=None, allow_failure=True)
        if retry is not None:
            best_result = retry
            if retry.text.strip():
                return retry, attempts, time.perf_counter() - started_at

        fallback_model = _derive_non_thinking_model(self.config.api.model)
        if fallback_model:
            fallback = invoke(strategy="empty_text_fallback", model_override=fallback_model, allow_failure=True)
            if fallback is not None:
                best_result = fallback
                if fallback.text.strip():
                    return fallback, attempts, time.perf_counter() - started_at

        return best_result, attempts, time.perf_counter() - started_at

    def _complete_with_pipeline_retries(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]], float, list[dict[str, Any]]]:
        started_at = time.perf_counter()
        pipeline_failures: list[dict[str, Any]] = []
        max_pipeline_attempts = 2 if int(self.config.api.max_retries) > 0 else 1
        last_exc: Exception | None = None
        sample_timeout = max(1.0, float(self.config.api.sample_timeout_s))
        deadline = started_at + sample_timeout

        for pipeline_attempt in range(1, max_pipeline_attempts + 1):
            if _remaining_timeout_s(deadline) <= 0:
                timeout_exc = TimeoutError(
                    f"Sample-level timeout exceeded after {sample_timeout:.1f}s."
                )
                pipeline_failures.append(
                    {
                        "pipeline_attempt": pipeline_attempt,
                        "error_type": type(timeout_exc).__name__,
                        "error": str(timeout_exc),
                    }
                )
                raise timeout_exc
            try:
                result, completion_attempts, _ = self._complete_with_recovery(messages, deadline=deadline)
                return result, completion_attempts, time.perf_counter() - started_at, pipeline_failures
            except Exception as exc:
                last_exc = exc
                pipeline_failures.append(
                    {
                        "pipeline_attempt": pipeline_attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if pipeline_attempt >= max_pipeline_attempts or not _is_retryable_inference_error(exc):
                    raise
                sleep_s = min(12.0, 2.0 * pipeline_attempt)
                remaining_timeout = _remaining_timeout_s(deadline)
                if remaining_timeout is not None and remaining_timeout <= 0:
                    raise
                if remaining_timeout is not None:
                    sleep_s = min(sleep_s, remaining_timeout)
                time.sleep(sleep_s)

        assert last_exc is not None
        raise last_exc

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

    def _summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if self._is_proactive_mode():
            return summarize_proactive(rows, thresholds=self.config.evaluation.iou_thresholds)
        return summarize_vqa(rows)

    def load_examples(self) -> list[EgoGazeVQASample]:
        examples = self.dataset.load_examples(self.config.data.split)
        return select_egogazevqa_examples(
            examples=examples,
            limit=self.config.data.limit,
            seed=self.config.data.seed,
            manifest_entries=self._manifest_entries(),
        )

    def sample_manifest(self, per_task: int, output_path: Path) -> list[dict[str, Any]]:
        examples = self.dataset.load_examples(self.config.data.split)
        manifest = build_egogazevqa_manifest(examples, per_group=per_task, seed=self.config.data.seed)
        write_jsonl(output_path, manifest)
        return manifest

    def inspect(self) -> dict[str, Any]:
        stats = self.dataset.inspect(self.config.data.split)
        stats["dataset_dir"] = str(self.config.data.dataset_dir)
        stats["metadata_path"] = str(self.dataset.resolve_metadata_path())
        stats["task_mode"] = "proactive" if self._is_proactive_mode() else "qa"
        return stats

    def evaluate_existing(self) -> dict[str, Any]:
        results_path = self.config.experiment.output_dir / "results.jsonl"
        rows = read_jsonl(results_path)
        summary = self._summarize(rows)
        dump_json(self.config.experiment.output_dir / "metrics.json", summary)
        return summary

    def run(self) -> dict[str, Any]:
        output_dir = ensure_dir(self.config.experiment.output_dir)
        results_path = output_dir / "results.jsonl"
        frame_cache_dir = ensure_dir(self.config.data.cache_dir or (output_dir / "_decoded_frames"))
        visuals_dir = ensure_dir(output_dir / "visuals") if self.config.evaluation.export_visualizations else output_dir / "visuals"
        dump_json(output_dir / "resolved_config.json", self.config.as_dict())

        if not self.config.experiment.resume and results_path.exists():
            results_path.unlink()

        completed_keys = set()
        if self.config.experiment.resume and results_path.exists():
            completed_keys = {row["example"]["sample_id"] for row in read_jsonl(results_path)}

        if not self.config.experiment.dry_run:
            self._ensure_client()

        examples = self.load_examples()
        rows: list[dict[str, Any]] = read_jsonl(results_path) if self.config.experiment.resume and results_path.exists() else []
        visuals_written = sum(1 for row in rows if row.get("visual_written"))
        for example in tqdm(examples, desc="Running EgoGazeVQA"):
            if example.sample_id in completed_keys:
                continue
            row = self._run_single(example, frame_cache_dir, visuals_dir, visuals_written)
            rows.append(row)
            append_jsonl(results_path, row)
            if row.get("visual_written"):
                visuals_written += 1

        summary = self._summarize(rows)
        dump_json(output_dir / "metrics.json", summary)
        return {"output_dir": str(output_dir), "results_path": str(results_path), "metrics": summary}

    def _box_format_prompt(self) -> str:
        fmt = self.config.prompt.prediction_box_format
        if fmt == "yxyx":
            return "ymin, xmin, ymax, xmax"
        if fmt == "xywh":
            return "xmin, ymin, width, height"
        return "xmin, ymin, xmax, ymax"

    def _build_system_prompt(self, include_gaze: bool) -> str:
        if self._is_proactive_mode():
            if self._is_fast_mode():
                lines = [
                    "You are an Ego-Pilot assistant for fast proactive egocentric intent reasoning.",
                    "Use the recent frames and the latest gaze coordinate to infer the best answer option.",
                    "The last image is the current frame.",
                    "Return exactly five short lines and nothing else.",
                    "Reasoning: <one short sentence>",
                    "Intention: <one short sentence>",
                    "Answer: <single option letter>",
                    "Object Class: <short noun phrase>",
                    f"Localization: <box>[[{self._box_format_prompt()}]]</box>",
                    "Use absolute pixel coordinates for the latest frame only.",
                    "The box must cover the visible target object in the latest frame, not the whole scene region.",
                    "Prefer a tight object box. Do not box empty context, large background regions, or the entire container unless the container itself is the target.",
                    "When gaze is provided, choose the target object nearest to the latest gaze point that best explains the answer.",
                    "If possible, make the gaze point fall inside the box.",
                    "Keep the whole response under 80 words.",
                ]
                if not self.config.prompt.allow_silence:
                    lines.append("Do not output <SILENCE>. Always choose the best option and include a box.")
                elif include_gaze:
                    lines.append(
                        f"If no actionable intent is detectable, output only the exact token {self.config.prompt.silence_token}."
                    )
                return "\n".join(lines)

            lines = [
                "You are an Ego-Pilot assistant for proactive egocentric intent reasoning.",
                (
                    f"You will receive chronologically ordered frames from roughly the past "
                    f"{self.config.prompt.window_seconds} seconds of a first-person video clip."
                ),
                "The last attached image is the current frame.",
                "Infer the wearer's immediate implicit intent and localize the best target object in the latest frame.",
                "Think step by step about scene context, handled objects, and short-term temporal changes.",
            ]
            if include_gaze:
                lines.append("Use the latest gaze coordinate as a physical anchor for your reasoning whenever it is provided.")
            lines.extend(
                [
                    (
                        f"If no actionable intent is detectable, output only the exact token "
                        f"{self.config.prompt.silence_token}."
                        if self.config.prompt.allow_silence
                        else "Always choose the single best answer option from the candidates."
                    ),
                    "Output exactly these five fields and nothing else:",
                    "Reasoning: <brief step-by-step causal reasoning>",
                    "Intention: <one-sentence latent need>",
                    "Answer: <single option letter>",
                    "Object Class: <target object category>",
                    f"Localization: <box>[[{self._box_format_prompt()}]]</box>",
                    "Use absolute pixel coordinates for the latest frame only.",
                    "Never output normalized coordinates in the 0-1 range.",
                    "Localize the specific visible target object rather than a broad scene region.",
                    "When gaze is available, prefer the object nearest to the latest gaze point that causally supports the chosen answer.",
                    "Make the box tight and include the gaze point whenever that point lies on the target object.",
                ]
            )
            return "\n".join(lines)

        lines = [
            "You are an expert assistant for egocentric video question answering.",
            "You will receive chronologically ordered frames from a short first-person video clip.",
            "Answer the multiple-choice question using the visual evidence.",
            "Return only the single option letter, such as A or B.",
        ]
        if include_gaze:
            lines.append("Additional normalized gaze coordinates indicate where the wearer is attending in selected frames.")
        return "\n".join(lines)

    def _resolve_gaze_anchor(self, sample: EgoGazeVQASample, frames: list[VideoFrame]) -> GazeAnchor | None:
        if not frames or not sample.gaze_sequence:
            return None
        latest_frame = frames[-1]
        nearest = min(sample.gaze_sequence, key=lambda point: abs(point.frame - latest_frame.source_frame))
        if nearest.x is None or nearest.y is None:
            return None

        width, height = _frame_size(latest_frame.path)
        x_norm = min(1.0, max(0.0, float(nearest.x)))
        y_norm = min(1.0, max(0.0, float(nearest.y)))
        x_px = min(max(0.0, x_norm * width), max(0, width - 1))
        y_px = min(max(0.0, y_norm * height), max(0, height - 1))
        ratio = max(0.05, float(self.config.evaluation.proxy_box_size_ratio))
        proxy_box = Box.from_center(x_px, y_px, width * ratio, height * ratio).clip(width, height)
        confidence = float(nearest.confidence) if nearest.confidence is not None else None
        return GazeAnchor(
            source_frame=latest_frame.source_frame,
            x_norm=x_norm,
            y_norm=y_norm,
            confidence=confidence,
            width=width,
            height=height,
            x_px=x_px,
            y_px=y_px,
            proxy_box=proxy_box,
            frame_path=latest_frame.path,
        )

    def _format_qa_gaze_text(self, sample: EgoGazeVQASample, frames: list[VideoFrame]) -> str:
        if not sample.gaze_sequence:
            return "No gaze annotations are available for this clip."
        max_points = max(1, self.config.prompt.max_gaze_points)
        lines = [
            "Gaze guidance:",
            "Coordinates are normalized. x increases from left to right; y increases from top to bottom.",
        ]
        for frame in frames:
            nearest = min(sample.gaze_sequence, key=lambda point: abs(point.frame - frame.source_frame))
            lines.append(
                f"Frame source={frame.source_frame}: gaze=({nearest.x}, {nearest.y}), confidence={nearest.confidence}"
            )
            if len(lines) - 2 >= max_points:
                break
        return "\n".join(lines)

    def _build_qa_messages(self, sample: EgoGazeVQASample, frames: list[VideoFrame]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        include_gaze = self.config.prompt.include_gaze_text or "gaze" in self.config.prompt.variant
        system_prompt = self._build_system_prompt(include_gaze=include_gaze)
        options_text = "\n".join(sample.answer_options)
        user_sections = [
            f"Question type: {sample.qa_type}",
            f"Dataset: {sample.dataset}",
            f"Question: {sample.question}",
            f"Options:\n{options_text}",
        ]
        if include_gaze:
            user_sections.append(self._format_qa_gaze_text(sample, frames))
        user_sections.append("Return only the correct option letter.")
        user_prompt = "\n\n".join(user_sections)

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for index, frame in enumerate(frames, start=1):
            content.append({"type": "text", "text": f"Frame {index} (source_frame={frame.source_frame})"})
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(frame.path, self.config.prompt.image_max_pixels)}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return messages, {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "attached_frames": [str(frame.path) for frame in frames],
        }

    def _build_proactive_messages(
        self,
        sample: EgoGazeVQASample,
        frames: list[VideoFrame],
        gaze_anchor: GazeAnchor | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        include_gaze = self.config.prompt.include_gaze_text
        system_prompt = self._build_system_prompt(include_gaze=include_gaze)
        options_text = "\n".join(sample.answer_options)
        latest_width, latest_height = _frame_size(frames[-1].path) if frames else (0, 0)
        if self._is_fast_mode():
            user_sections = [
                f"Question: {sample.question}",
                f"Options:\n{options_text}",
                f"Latest frame size: {latest_width}x{latest_height}.",
                "Choose the best answer, infer the short intent, and localize the most likely target in the latest frame.",
                "Keep every field brief and complete all five fields.",
                "Do not return 0-1 normalized box coordinates.",
                "The localization must be a tight box around one visible target object.",
                "Prefer the object being attended right now, not a loose area around it.",
            ]
        else:
            user_sections = [
                "Task: gaze-to-intent proactive reasoning on egocentric video.",
                (
                    f"The attached frames cover the recent ~{self.config.prompt.window_seconds} seconds. "
                    "The last image is the current frame."
                ),
                f"Question type: {sample.qa_type}",
                f"Question: {sample.question}",
                f"Options:\n{options_text}",
                "Infer the wearer's latent short-term need before mapping it to the best answer option.",
                "Then localize the most relevant target object in the latest frame.",
                f"The latest frame size is {latest_width}x{latest_height}. Output absolute pixel coordinates for this frame only.",
                "Do not return 0-1 normalized box coordinates.",
                "Return one tight box around the specific visible target object, not around a broad region.",
            ]
        if include_gaze:
            if gaze_anchor is None:
                user_sections.append("Current gaze coordinates are unavailable for this clip.")
            else:
                gaze_line = (
                    f"Current gaze on the latest frame: [{gaze_anchor.x_norm:.4f}, {gaze_anchor.y_norm:.4f}] "
                    f"(confidence={gaze_anchor.confidence})."
                )
                user_sections.append(gaze_line)
                user_sections.append(
                    f"Current gaze in pixels on the latest frame: ({int(round(gaze_anchor.x_px))}, {int(round(gaze_anchor.y_px))})."
                )
                user_sections.append("Treat the gaze coordinate as a physical anchor, not as the final answer by itself.")
                user_sections.append(
                    "Among plausible objects, prefer the visible object whose surface or center is closest to the latest gaze point."
                )
                user_sections.append(
                    "If the wearer is looking at the target itself, the gaze point should lie inside or immediately adjacent to the predicted box."
                )
        if self.config.prompt.allow_silence:
            user_sections.append(
                f"If no actionable intent can be inferred, return only {self.config.prompt.silence_token}."
            )
        else:
            user_sections.append(
                "Do not output <SILENCE>. Even if evidence is weak, choose the best answer option and localize the most likely target."
            )
        user_prompt = "\n\n".join(user_sections)

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for index, frame in enumerate(frames, start=1):
            label = "Current frame" if index == len(frames) else f"Past frame {index}"
            content.append({"type": "text", "text": f"{label} (source_frame={frame.source_frame})"})
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(frame.path, self.config.prompt.image_max_pixels)}})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        prompt_record: dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "attached_frames": [str(frame.path) for frame in frames],
        }
        if gaze_anchor is not None:
            prompt_record["gaze_anchor"] = gaze_anchor.to_dict()
        return messages, prompt_record

    def _build_messages(
        self,
        sample: EgoGazeVQASample,
        frames: list[VideoFrame],
        gaze_anchor: GazeAnchor | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self._is_proactive_mode():
            return self._build_proactive_messages(sample, frames, gaze_anchor)
        return self._build_qa_messages(sample, frames)

    def _evaluate_proactive_prediction(
        self,
        sample: EgoGazeVQASample,
        frames: list[VideoFrame],
        prediction: Prediction,
        predicted_answer: str | None,
        gaze_anchor: GazeAnchor | None,
    ) -> dict[str, Any]:
        latest_frame_path = frames[-1].path if frames else None
        latest_width, latest_height = _frame_size(latest_frame_path) if latest_frame_path else (0, 0)
        raw_box = _canonicalize_box(prediction.box) if prediction.box is not None else None
        model_box, box_coordinate_space = _coerce_box_to_frame(raw_box, latest_width, latest_height)
        clipped_box, box_refinement = _refine_box_with_gaze(model_box, gaze_anchor, latest_width, latest_height)
        prediction_payload = prediction.to_dict()
        prediction_payload["box_xyxy_raw"] = raw_box.as_list() if raw_box else None
        prediction_payload["box_coordinate_space"] = box_coordinate_space
        prediction_payload["box_xyxy_model"] = model_box.as_list() if model_box else None
        prediction_payload["box_refinement"] = box_refinement
        prediction_payload["box_xyxy"] = clipped_box.as_list() if clipped_box else None
        raw_alignment = _box_alignment_metrics(model_box, gaze_anchor.to_dict() if gaze_anchor else None)
        refined_alignment = _box_alignment_metrics(clipped_box, gaze_anchor.to_dict() if gaze_anchor else None)
        reference_intention = sample.answer_option_map.get(sample.correct_answer)
        intent_similarity = _intent_similarity(prediction.intention, reference_intention)

        result: dict[str, Any] = {
            "prediction": prediction_payload,
            "predicted_answer": predicted_answer,
            "reference_answer": sample.correct_answer,
            "reference_intention": reference_intention,
            "correct": predicted_answer == sample.correct_answer,
            "intent_consistent": bool(prediction.intention) and (
                predicted_answer == sample.correct_answer or _intent_match(prediction.intention, reference_intention)
            ),
            "strict_intent_match": _intent_match(prediction.intention, reference_intention),
            "intent_match": _intent_match(prediction.intention, reference_intention),
            "intent_similarity": intent_similarity,
            "is_silence": prediction.is_silence,
            "has_box": refined_alignment["has_box"],
            "raw_has_box": raw_alignment["has_box"],
            "gaze_anchor": gaze_anchor.to_dict() if gaze_anchor else None,
            "latest_frame": str(latest_frame_path) if latest_frame_path else None,
            "proxy_iou": refined_alignment["proxy_iou"],
            "point_hit": refined_alignment["point_hit"],
            "center_distance_norm": refined_alignment["center_distance_norm"],
            "raw_proxy_iou": raw_alignment["proxy_iou"],
            "raw_point_hit": raw_alignment["point_hit"],
            "raw_center_distance_norm": raw_alignment["center_distance_norm"],
            "refinement_applied": bool(box_refinement),
        }

        for threshold in self.config.evaluation.iou_thresholds:
            result[f"proxy_iou_ge_{str(threshold).replace('.', '_')}"] = False
            result[f"raw_proxy_iou_ge_{str(threshold).replace('.', '_')}"] = False

        if clipped_box is None or gaze_anchor is None:
            for threshold in self.config.evaluation.iou_thresholds:
                if raw_alignment["proxy_iou"] is not None:
                    result[f"raw_proxy_iou_ge_{str(threshold).replace('.', '_')}"] = raw_alignment["proxy_iou"] >= threshold
            return result

        result["proxy_gt_box_xyxy"] = gaze_anchor.proxy_box.as_list()
        for threshold in self.config.evaluation.iou_thresholds:
            if refined_alignment["proxy_iou"] is not None:
                result[f"proxy_iou_ge_{str(threshold).replace('.', '_')}"] = refined_alignment["proxy_iou"] >= threshold
            if raw_alignment["proxy_iou"] is not None:
                result[f"raw_proxy_iou_ge_{str(threshold).replace('.', '_')}"] = raw_alignment["proxy_iou"] >= threshold
        return result

    def _maybe_write_visual(
        self,
        row: dict[str, Any],
        visuals_dir: Path,
        visuals_written: int,
        frames: list[VideoFrame],
        gaze_anchor: GazeAnchor | None,
    ) -> bool:
        if (
            not self.config.evaluation.export_visualizations
            or visuals_written >= self.config.evaluation.visualization_limit
            or not self._is_proactive_mode()
            or row.get("status") != "completed"
        ):
            return False
        if not frames or gaze_anchor is None:
            return False
        prediction = row.get("prediction") or {}
        box_xyxy = prediction.get("box_xyxy")
        if not box_xyxy:
            return False
        raw_box_xyxy = prediction.get("box_xyxy_model")

        pred_box = Box.from_sequence(box_xyxy, fmt="xyxy")
        raw_pred_box = None
        if isinstance(raw_box_xyxy, list) and len(raw_box_xyxy) == 4:
            raw_pred_box = Box.from_sequence(raw_box_xyxy, fmt="xyxy")
        caption = (
            f"{row['example']['sample_id']} ans={row.get('predicted_answer')} "
            f"gt={row.get('reference_answer')} raw_iou={row.get('raw_proxy_iou')} iou={row.get('proxy_iou')}"
        )
        file_stem = _safe_filename(str(row["example"]["sample_id"]))
        draw_boxes(
            image_path=frames[-1].path,
            gt_boxes=[gaze_anchor.proxy_box],
            pred_box=pred_box,
            raw_pred_box=raw_pred_box,
            output_path=visuals_dir / f"{file_stem}.jpg",
            caption=caption,
            gaze_point=(gaze_anchor.x_px, gaze_anchor.y_px),
        )
        return True

    def _run_single(
        self,
        sample: EgoGazeVQASample,
        frame_cache_dir: Path,
        visuals_dir: Path,
        visuals_written: int,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "timestamp": utc_timestamp(),
            "status": "pending",
            "dataset": sample.dataset,
            "qa_type": sample.qa_type,
            "variant": self.config.prompt.variant,
            "task_mode": "proactive" if self._is_proactive_mode() else "qa",
            "example": sample.to_dict(),
            "frames": [],
            "visual_written": False,
        }

        try:
            frames = decode_sampled_frames(
                sample,
                max_frames=self.config.prompt.max_frames,
                cache_dir=frame_cache_dir,
                window_seconds=self.config.prompt.window_seconds if self._is_proactive_mode() else None,
                source_fps=self.config.data.native_fps,
                sample_fps=self.config.prompt.sample_fps,
                tail_only=self._is_proactive_mode(),
            )
            row["frames"] = [str(frame.path) for frame in frames]
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            return row

        if not frames and self.config.data.skip_missing_images:
            row["status"] = "skipped_missing_videos"
            return row

        gaze_anchor = self._resolve_gaze_anchor(sample, frames) if self._is_proactive_mode() else None
        messages, prompt_record = self._build_messages(sample, frames, gaze_anchor)
        if self.config.experiment.save_prompts:
            row["prompt"] = prompt_record

        try:
            if self.config.experiment.dry_run:
                row["status"] = "dry_run"
                if gaze_anchor is not None:
                    row["gaze_anchor"] = gaze_anchor.to_dict()
                return row

            result, completion_attempts, total_latency_s, pipeline_failures = self._complete_with_pipeline_retries(messages)

            if self._is_proactive_mode():
                prediction = parse_prediction(
                    result.text,
                    silence_token=self.config.prompt.silence_token,
                    box_format=self.config.prompt.prediction_box_format,
                )
                predicted_answer = extract_answer_letter(result.text, sample.answer_option_map)
                row["status"] = "completed"
                row.update(
                    self._evaluate_proactive_prediction(
                        sample=sample,
                        frames=frames,
                        prediction=prediction,
                        predicted_answer=predicted_answer,
                        gaze_anchor=gaze_anchor,
                    )
                )
            else:
                predicted_answer = extract_answer_letter(result.text, sample.answer_option_map)
                prediction = VQAPrediction(raw_text=result.text, answer=predicted_answer)
                row["status"] = "completed"
                row["prediction"] = prediction.to_dict()
                row["predicted_answer"] = predicted_answer
                row["reference_answer"] = sample.correct_answer
                row["correct"] = predicted_answer == sample.correct_answer

            row["response"] = {
                "text": result.text,
                "latency_s": total_latency_s,
                "backend_latency_s": result.latency_s,
                "usage": result.usage,
                "finish_reason": result.finish_reason,
                "model": result.model,
                "attempt_count": len(completion_attempts),
                "recovery_attempts": completion_attempts,
                "pipeline_failures": pipeline_failures,
                "empty_text_recovered": bool(result.text.strip()) and len(completion_attempts) > 1,
            }
            try:
                row["visual_written"] = self._maybe_write_visual(row, visuals_dir, visuals_written, frames, gaze_anchor)
            except Exception as exc:
                row["visual_written"] = False
                row["visual_error"] = str(exc)
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)

        return row
