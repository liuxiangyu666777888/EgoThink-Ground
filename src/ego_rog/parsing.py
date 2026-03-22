from __future__ import annotations

import re
from dataclasses import dataclass

from .geometry import Box


BOX_TAG_RE = re.compile(r"<box>\s*\[\[?\s*([^\]]+?)\s*\]?\]\s*</box>", re.IGNORECASE | re.DOTALL)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Prediction:
    raw_text: str
    is_silence: bool
    reasoning: str | None
    intention: str | None
    object_class: str | None
    box: Box | None

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "is_silence": self.is_silence,
            "reasoning": self.reasoning,
            "intention": self.intention,
            "object_class": self.object_class,
            "box_xyxy": self.box.as_list() if self.box else None,
        }


def _extract_field(text: str, labels: list[str]) -> str | None:
    for label in labels:
        pattern = re.compile(
            rf"^{re.escape(label)}\s*:\s*(.+?)(?=^\w[\w ]*\s*:|\Z)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _parse_box(text: str, box_format: str) -> Box | None:
    match = BOX_TAG_RE.search(text)
    candidate = match.group(1) if match else None
    if candidate is None:
        candidate = _extract_field(text, ["Localization", "Box"])
    if candidate is None:
        return None
    values = [float(value) for value in NUMBER_RE.findall(candidate)]
    if len(values) < 4:
        return None
    return Box.from_sequence(values[:4], fmt=box_format)


def parse_prediction(text: str, silence_token: str, box_format: str) -> Prediction:
    stripped = text.strip()
    if stripped == silence_token or stripped.upper() == silence_token.upper():
        return Prediction(raw_text=text, is_silence=True, reasoning=None, intention=None, object_class=None, box=None)

    return Prediction(
        raw_text=text,
        is_silence=False,
        reasoning=_extract_field(stripped, ["Reasoning"]),
        intention=_extract_field(stripped, ["Intention"]),
        object_class=_extract_field(stripped, ["Object Class", "Object"]),
        box=_parse_box(stripped, box_format),
    )
