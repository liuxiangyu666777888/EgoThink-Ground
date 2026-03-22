from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "Box":
        return cls(float(x), float(y), float(x) + float(w), float(y) + float(h))

    @classmethod
    def from_center(cls, cx: float, cy: float, width: float, height: float) -> "Box":
        half_w = float(width) / 2.0
        half_h = float(height) / 2.0
        return cls(float(cx) - half_w, float(cy) - half_h, float(cx) + half_w, float(cy) + half_h)

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, float, float, float], fmt: str) -> "Box":
        if len(values) != 4:
            raise ValueError(f"Expected 4 values, got {len(values)}")
        a, b, c, d = [float(v) for v in values]
        if fmt == "xyxy":
            return cls(a, b, c, d)
        if fmt == "xywh":
            return cls.from_xywh(a, b, c, d)
        if fmt == "yxyx":
            return cls(b, a, d, c)
        raise ValueError(f"Unsupported box format: {fmt}")

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            xmin=max(0.0, min(float(width), self.xmin)),
            ymin=max(0.0, min(float(height), self.ymin)),
            xmax=max(0.0, min(float(width), self.xmax)),
            ymax=max(0.0, min(float(height), self.ymax)),
        )

    def as_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= float(x) <= self.xmax and self.ymin <= float(y) <= self.ymax

    def iou(self, other: "Box") -> float:
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)
        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        union = self.area + other.area - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
