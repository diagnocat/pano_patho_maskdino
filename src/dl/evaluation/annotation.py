from __future__ import annotations

from typing import TypedDict

import numpy as np
from typing_extensions import NotRequired


class InstanceAnnotation(TypedDict):
    category_id: int
    bbox: tuple[float, float, float, float]  # xyxy abs
    mask_rle: str  # RLE as a string
    tags: dict[str, int]
    score: float
    mask: NotRequired[np.ndarray]


class ImageAnnotation(TypedDict):
    instance_annotations: list[InstanceAnnotation]
    width: int
    height: int
    image_id: int
