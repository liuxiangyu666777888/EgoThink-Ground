from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ego_rog.geometry import Box
from src.ego_rog.utils import ensure_dir
from src.ego_rog.visualization import draw_boxes


def _maybe_box(values: object) -> Box | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    return Box.from_sequence([float(value) for value in values], fmt="xyxy")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:180]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export EgoGazeVQA visualization overlays from results.jsonl")
    parser.add_argument("--results", required=True, help="Path to results.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered images")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of visuals to export")
    parser.add_argument(
        "--sort-by",
        choices=("order", "proxy_iou", "improvement"),
        default="improvement",
        help="How to choose examples for export",
    )
    args = parser.parse_args()

    results_path = Path(args.results).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    rows: list[dict] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "completed":
                continue
            rows.append(row)

    if args.sort_by == "proxy_iou":
        rows.sort(key=lambda row: float(row.get("proxy_iou") or 0.0), reverse=True)
    elif args.sort_by == "improvement":
        rows.sort(
            key=lambda row: float(row.get("proxy_iou") or 0.0) - float(row.get("raw_proxy_iou") or 0.0),
            reverse=True,
        )

    exported = 0
    for row in rows:
        if exported >= args.limit:
            break
        image_path = Path(row.get("latest_frame") or "")
        if not image_path.exists():
            continue
        gaze_anchor = row.get("gaze_anchor") or {}
        gt_box = _maybe_box(row.get("proxy_gt_box_xyxy") or gaze_anchor.get("proxy_box_xyxy"))
        pred_box = _maybe_box((row.get("prediction") or {}).get("box_xyxy"))
        raw_pred_box = _maybe_box((row.get("prediction") or {}).get("box_xyxy_model"))
        raw_iou = row.get("raw_proxy_iou")
        refined_iou = row.get("proxy_iou")
        if raw_iou is None and raw_pred_box is not None and gt_box is not None:
            raw_iou = raw_pred_box.iou(gt_box)
        if refined_iou is None and pred_box is not None and gt_box is not None:
            refined_iou = pred_box.iou(gt_box)
        gaze_point = None
        if gaze_anchor.get("x_px") is not None and gaze_anchor.get("y_px") is not None:
            gaze_point = (float(gaze_anchor["x_px"]), float(gaze_anchor["y_px"]))
        caption = (
            f"{row['example']['sample_id']} ans={row.get('predicted_answer')} gt={row.get('reference_answer')} "
            f"raw_iou={raw_iou} iou={refined_iou} refine={row.get('prediction', {}).get('box_refinement')}"
        )
        name = _safe_name(str(row["example"]["sample_id"])) + ".jpg"
        draw_boxes(
            image_path=image_path,
            gt_boxes=[gt_box] if gt_box else [],
            pred_box=pred_box,
            raw_pred_box=raw_pred_box,
            output_path=output_dir / name,
            caption=caption,
            gaze_point=gaze_point,
        )
        exported += 1

    print({"results": str(results_path), "output_dir": str(output_dir), "exported": exported})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
