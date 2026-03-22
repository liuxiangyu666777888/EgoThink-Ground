from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .geometry import Box


def draw_boxes(
    image_path: Path,
    gt_boxes: list[Box],
    pred_box: Box | None,
    raw_pred_box: Box | None,
    output_path: Path,
    caption: str,
    gaze_point: tuple[float, float] | None = None,
) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in gt_boxes:
            draw.rectangle(box.as_list(), outline="lime", width=4)
        if raw_pred_box is not None:
            draw.rectangle(raw_pred_box.as_list(), outline="yellow", width=3)
        if pred_box is not None:
            draw.rectangle(pred_box.as_list(), outline="red", width=4)
        if gaze_point is not None:
            x, y = gaze_point
            radius = max(4, int(min(image.width, image.height) * 0.01))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="cyan", width=3)
        draw.rectangle((0, 0, image.width, 50), fill="black")
        draw.text((10, 14), caption[:180], fill="white")
        image.save(output_path)
