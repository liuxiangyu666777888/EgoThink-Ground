from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data import EgoIntentionExample


@dataclass
class FrameRef:
    frame_index: int
    path: Path
    offset_seconds: float


@dataclass
class FrameWindow:
    current_path: Path | None
    frames: list[FrameRef]
    missing_indices: list[int]
    mode: str

    @property
    def exists(self) -> bool:
        return self.current_path is not None and self.current_path.exists()


class FrameLocator:
    def __init__(self, frame_roots: list[Path], path_replacements: dict[str, str] | None = None):
        self.frame_roots = frame_roots
        self.path_replacements = path_replacements or {}

    def _resolve_direct_path(self, image_url: str, clip_id: str, frame_index: int) -> Path | None:
        basename = Path(image_url).stem
        expected = f"{clip_id}_{frame_index:06d}"
        if basename != expected:
            return None
        for prefix, replacement in self.path_replacements.items():
            if image_url.startswith(prefix):
                mapped = Path(image_url.replace(prefix, replacement, 1))
                if mapped.exists():
                    return mapped
        direct = Path(image_url)
        return direct if direct.exists() else None

    def _candidate_paths(self, clip_id: str, frame_index: int, suffix: str) -> list[Path]:
        suffixes = [suffix]
        if suffix.lower() == ".jpeg":
            suffixes.append(".jpg")
        elif suffix.lower() == ".jpg":
            suffixes.append(".jpeg")
        candidates: list[Path] = []
        for item_suffix in suffixes:
            basename = f"{clip_id}_{frame_index:06d}{item_suffix}"
            for root in self.frame_roots:
                candidates.extend(
                    [
                        root / basename,
                        root / clip_id / basename,
                        root / "paco_frames" / basename,
                        root / "paco_frames" / clip_id / basename,
                    ]
                )
        return candidates

    def resolve_frame(self, image_url: str, clip_id: str, frame_index: int, suffix: str = ".jpeg") -> Path | None:
        direct = self._resolve_direct_path(image_url, clip_id, frame_index)
        if direct is not None:
            return direct
        for candidate in self._candidate_paths(clip_id, frame_index, suffix):
            if candidate.exists():
                return candidate
        return None

    def build_window(
        self,
        example: EgoIntentionExample,
        mode: str,
        window_seconds: int,
        sample_fps: int,
        source_fps: int,
        max_frames: int,
        allow_sparse: bool,
    ) -> FrameWindow:
        suffix = Path(example.image_url).suffix or ".jpeg"
        current_path = self.resolve_frame(example.image_url, example.clip_id, example.frame_index, suffix=suffix)
        if mode == "single_frame":
            frames = [FrameRef(example.frame_index, current_path, 0.0)] if current_path else []
            missing = [] if current_path else [example.frame_index]
            return FrameWindow(current_path=current_path, frames=frames, missing_indices=missing, mode=mode)

        step = max(int(round(source_fps / max(sample_fps, 1))), 1)
        frame_budget = max(1, min(max_frames, window_seconds * max(sample_fps, 1)))

        frames: list[FrameRef] = []
        missing_indices: list[int] = []
        for offset in reversed(range(frame_budget)):
            frame_index = max(0, example.frame_index - offset * step)
            path = self.resolve_frame(example.image_url, example.clip_id, frame_index, suffix=suffix)
            if path is None:
                missing_indices.append(frame_index)
                if not allow_sparse:
                    return FrameWindow(current_path=current_path, frames=[], missing_indices=missing_indices, mode=mode)
                continue
            frames.append(
                FrameRef(
                    frame_index=frame_index,
                    path=path,
                    offset_seconds=(example.frame_index - frame_index) / max(source_fps, 1),
                )
            )

        return FrameWindow(current_path=current_path, frames=frames, missing_indices=missing_indices, mode=mode)
