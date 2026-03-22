from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .geometry import Box
from .utils import load_json


FILENAME_RE = re.compile(r"(?P<clip_id>.+?)_(?P<frame_index>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


@dataclass
class EgoIntentionExample:
    sample_id: str
    split: str
    task: str
    query: str
    image_url: str
    width: int
    height: int
    gt_objects: list[str]
    gt_boxes: list[Box]
    clip_id: str
    frame_index: int

    def key(self) -> str:
        return f"{self.sample_id}:{self.task}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "task": self.task,
            "query": self.query,
            "image_url": self.image_url,
            "width": self.width,
            "height": self.height,
            "gt_objects": self.gt_objects,
            "gt_boxes_xyxy": [box.as_list() for box in self.gt_boxes],
            "clip_id": self.clip_id,
            "frame_index": self.frame_index,
        }


@dataclass
class PromptAssets:
    primary_function: dict[str, str]
    context_icl: dict[str, list[str]]
    uncommon_icl: dict[str, list[str]]


def _parse_filename(image_url: str) -> tuple[str, int]:
    match = FILENAME_RE.match(Path(image_url).name)
    if not match:
        raise ValueError(f"Unsupported image filename: {image_url}")
    return match.group("clip_id"), int(match.group("frame_index"))


def _parse_box_text(raw_box: Any) -> list[float]:
    if isinstance(raw_box, str):
        parsed = ast.literal_eval(raw_box)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Unsupported bbox string: {raw_box}")
        return [float(value) for value in parsed]
    if isinstance(raw_box, (list, tuple)):
        return [float(value) for value in raw_box]
    raise ValueError(f"Unsupported bbox type: {type(raw_box).__name__}")


def _flatten_box_dict(raw_box_dict: dict[str, Any]) -> tuple[list[str], list[Box]]:
    objects: list[str] = []
    boxes: list[Box] = []
    for object_name, box_group in raw_box_dict.items():
        if not isinstance(box_group, list):
            continue
        for raw_box in box_group:
            objects.append(object_name)
            boxes.append(Box.from_sequence(_parse_box_text(raw_box), fmt="xywh"))
    return objects, boxes


class EgoIntentionDataset:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def annotation_path(self, split: str) -> Path:
        return self.dataset_dir / f"egointention_{split}.json"

    def load_prompt_assets(self) -> PromptAssets:
        return PromptAssets(
            primary_function=load_json(self.dataset_dir / "primary_function.json"),
            context_icl=load_json(self.dataset_dir / "context_set_4_icl.json"),
            uncommon_icl=load_json(self.dataset_dir / "uncommon_set_4_icl.json"),
        )

    def load_examples(self, split: str, task: str = "both") -> list[EgoIntentionExample]:
        if task not in {"context", "uncommon", "both"}:
            raise ValueError(f"Unsupported task: {task}")
        task_list = ["context", "uncommon"] if task == "both" else [task]
        raw = load_json(self.annotation_path(split))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected annotation dict for split {split}")

        examples: list[EgoIntentionExample] = []
        for sample_id, sample in raw.items():
            clip_id, frame_index = _parse_filename(sample["image_url"])
            width = int(sample["width"])
            height = int(sample["height"])
            for task_name in task_list:
                if "context_bbox" in sample or "uncommon_box" in sample:
                    bbox_key = "context_bbox" if task_name == "context" else "uncommon_box"
                    query_key = "context_query" if task_name == "context" else "uncommon_query"
                    gt_objects, gt_boxes = _flatten_box_dict(sample.get(bbox_key, {}))
                else:
                    gt_objects = [str(sample["object"])]
                    gt_boxes = [Box.from_sequence(_parse_box_text(sample["bbox"]), fmt="xywh")]
                    query_key = "context_query" if task_name == "context" else "uncommon_query"
                examples.append(
                    EgoIntentionExample(
                        sample_id=str(sample_id),
                        split=split,
                        task=task_name,
                        query=str(sample.get(query_key, "")).strip(),
                        image_url=str(sample["image_url"]),
                        width=width,
                        height=height,
                        gt_objects=sorted(set(gt_objects)),
                        gt_boxes=[box.clip(width, height) for box in gt_boxes],
                        clip_id=clip_id,
                        frame_index=frame_index,
                    )
                )
        return examples

    def inventory(self) -> list[str]:
        objects: set[str] = set()
        for split in ("train", "val", "test"):
            path = self.annotation_path(split)
            if not path.exists():
                continue
            raw = load_json(path)
            for sample in raw.values():
                if "object" in sample:
                    objects.add(str(sample["object"]))
                objects.update(sample.get("context_bbox", {}).keys())
                objects.update(sample.get("uncommon_box", {}).keys())
        return sorted(objects)


def select_examples(
    examples: list[EgoIntentionExample],
    limit: int | None,
    seed: int,
    manifest_entries: list[dict[str, str]] | None = None,
) -> list[EgoIntentionExample]:
    if manifest_entries:
        index = {example.key(): example for example in examples}
        selected: list[EgoIntentionExample] = []
        for row in manifest_entries:
            key = f"{row['sample_id']}:{row['task']}"
            if key in index:
                selected.append(index[key])
        if limit is None or limit >= len(selected):
            return selected
        return selected[:limit]

    if limit is None or limit >= len(examples):
        return examples

    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    return pool[:limit]


def build_balanced_manifest(
    examples: Iterable[EgoIntentionExample],
    per_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[EgoIntentionExample]] = {"context": [], "uncommon": []}
    for example in examples:
        grouped.setdefault(example.task, []).append(example)

    manifest: list[dict[str, Any]] = []
    for task, task_examples in grouped.items():
        rng.shuffle(task_examples)
        for example in task_examples[:per_task]:
            manifest.append(
                {
                    "sample_id": example.sample_id,
                    "task": task,
                    "split": example.split,
                    "clip_id": example.clip_id,
                    "frame_index": example.frame_index,
                }
            )
    return manifest
