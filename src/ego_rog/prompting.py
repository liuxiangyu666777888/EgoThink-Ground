from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from .config import PromptConfig
from .data import EgoIntentionExample, PromptAssets
from .frames import FrameWindow


@dataclass
class PromptPackage:
    messages: list[dict]
    system_prompt: str
    user_prompt: str
    attached_frames: list[str]


def _image_to_data_url(path: Path, max_pixels: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_pixels and image.width * image.height > max_pixels:
            scale = math.sqrt(max_pixels / float(image.width * image.height))
            resized = (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            )
            image = image.resize(resized)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


class PromptBuilder:
    def __init__(self, config: PromptConfig, assets: PromptAssets, object_inventory: list[str]):
        self.config = config
        self.assets = assets
        self.object_inventory = object_inventory

    def _build_system_prompt(self) -> str:
        silence = self.config.silence_token
        lines = [
            "Role: Ego-Pilot, a first-person embodied assistant for egocentric visual intention grounding.",
            "Goal: infer whether the wearer has an immediate implicit need and localize the best target object in the latest frame.",
            "When no proactive intervention is necessary, output only the exact token <SILENCE>.",
            "If a target exists, output exactly four fields:",
            "Reasoning: <brief causal reasoning from recent observations>",
            "Intention: <one-sentence latent need>",
            "Object Class: <target object category>",
            "Localization: <box>[[xmin, ymin, xmax, ymax]]</box>",
            "The bounding box must refer to the latest frame only.",
        ]
        if self.config.variant == "rog_thinking":
            lines.extend(
                [
                    "Follow Reason-to-Ground in order:",
                    "1. Think over motion, hand state, gaze proxy and recent scene changes.",
                    "2. Infer the immediate latent intention.",
                    "3. Select the object that best satisfies the intention.",
                    "4. Localize the object in the latest frame.",
                ]
            )
        else:
            lines.append("Keep reasoning concise and prioritize accurate localization.")
        lines.append(f"Silence token: {silence}")
        return "\n".join(lines)

    def _build_inventory_block(self) -> str:
        if not self.config.include_object_inventory or not self.object_inventory:
            return ""
        inventory = ", ".join(self.object_inventory)
        return f"Candidate object classes from EgoIntention/PACO vocabulary:\n{inventory}"

    def _build_primary_function_block(self) -> str:
        if not self.config.include_primary_function or not self.assets.primary_function:
            return ""
        lines = [
            f"- {name}: {description}"
            for name, description in list(sorted(self.assets.primary_function.items()))[:25]
        ]
        return "Primary object functions (partial prior):\n" + "\n".join(lines)

    def _build_icl_block(self, task: str) -> str:
        count = max(self.config.icl_examples_per_prompt, 0)
        if count == 0:
            return ""
        source = self.assets.context_icl if task == "context" else self.assets.uncommon_icl
        examples: list[str] = []
        for values in source.values():
            examples.extend(values[:1])
            if len(examples) >= count:
                break
        if not examples:
            return ""
        return "Few-shot intention hints:\n" + "\n".join(f"- {example}" for example in examples[:count])

    def build(self, example: EgoIntentionExample, frame_window: FrameWindow) -> PromptPackage:
        system_prompt = self._build_system_prompt()
        sections = [
            f"Task type: {example.task}",
            f"User cue: {example.query}",
            "The attached frames are in chronological order; the last image is the latest frame.",
            "Reason about recent egocentric context before predicting the target object.",
        ]
        for block in (
            self._build_inventory_block(),
            self._build_primary_function_block(),
            self._build_icl_block(example.task),
        ):
            if block:
                sections.append(block)
        user_prompt = "\n\n".join(sections)

        content: list[dict] = [{"type": "text", "text": user_prompt}]
        attached_frames: list[str] = []
        for frame_ref in frame_window.frames:
            attached_frames.append(str(frame_ref.path))
            content.append(
                {
                    "type": "text",
                    "text": f"Frame t-{frame_ref.offset_seconds:.1f}s (frame_index={frame_ref.frame_index})",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _image_to_data_url(frame_ref.path, self.config.image_max_pixels),
                    },
                }
            )

        if not attached_frames:
            content.append(
                {
                    "type": "text",
                    "text": "No image could be resolved for this sample. If visual grounding is impossible, return <SILENCE>.",
                }
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return PromptPackage(
            messages=messages,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attached_frames=attached_frames,
        )
