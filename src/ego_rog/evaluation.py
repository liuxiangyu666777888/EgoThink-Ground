from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .data import EgoIntentionExample
from .parsing import Prediction


ALIASES = {
    "cell phone": "telephone",
    "phone": "telephone",
    "cellular telephone": "telephone",
    "mobile phone": "telephone",
}


def canonicalize_object_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.lower()
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return ALIASES.get(cleaned, cleaned)


def evaluate_prediction(
    example: EgoIntentionExample,
    prediction: Prediction,
    thresholds: list[float],
) -> dict[str, Any]:
    gt_names = [name for name in (canonicalize_object_name(item) for item in example.gt_objects) if name]
    pred_name = canonicalize_object_name(prediction.object_class)
    best_iou = 0.0
    if prediction.box is not None:
        best_iou = max((prediction.box.iou(gt_box) for gt_box in example.gt_boxes), default=0.0)

    result = {
        "best_iou": best_iou,
        "has_box": prediction.box is not None,
        "is_silence": prediction.is_silence,
        "object_match": pred_name in gt_names if pred_name else False,
    }
    for threshold in thresholds:
        result[f"iou_ge_{str(threshold).replace('.', '_')}"] = best_iou >= threshold
    return result


def summarize_results(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    def aggregate(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(group_rows)
        if total == 0:
            return {"count": 0}
        summary: dict[str, Any] = {
            "count": total,
            "valid_box_rate": sum(1 for row in group_rows if row.get("has_box")) / total,
            "silence_rate": sum(1 for row in group_rows if row.get("is_silence")) / total,
            "object_match_rate": sum(1 for row in group_rows if row.get("object_match")) / total,
            "mean_iou": sum(float(row.get("best_iou", 0.0)) for row in group_rows) / total,
            "completed_rate": sum(1 for row in group_rows if row.get("status") == "completed") / total,
        }
        for threshold in thresholds:
            key = f"iou_ge_{str(threshold).replace('.', '_')}"
            summary[f"p_at_{threshold}"] = sum(1 for row in group_rows if row.get(key)) / total
        return summary

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row.get("task", "unknown")].append(row)
        by_status[row.get("status", "unknown")].append(row)

    return {
        "overall": aggregate(rows),
        "by_task": {task: aggregate(task_rows) for task, task_rows in sorted(by_task.items())},
        "by_status": {status: aggregate(status_rows) for status, status_rows in sorted(by_status.items())},
    }
