"""YOLO annotation parsing and geometry helpers."""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_yolo_line(
    line: str, img_width: int, img_height: int
) -> Optional[Dict]:
    """Parse one line from a YOLO ``.txt`` label file.

    Detection format:   ``<cls> <xc> <yc> <w> <h>``  (normalised)
    Segmentation format: ``<cls> <x1> <y1> <x2> <y2> ...``  (normalised polygon)

    Returns a dict with keys:
    - ``class_id`` (int)
    - ``type`` — ``"bbox"`` or ``"polygon"``
    - ``bbox`` — ``[x, y, w, h]`` in absolute pixels (COCO convention)
    - ``area`` — annotation area in pixels²
    - ``polygon`` (only for segmentation) — list of ``(x, y)`` absolute tuples

    Returns ``None`` for malformed lines.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except (ValueError, IndexError):
        return None

    if len(coords) == 4:
        xc, yc, w, h = coords
        abs_x = (xc - w / 2) * img_width
        abs_y = (yc - h / 2) * img_height
        abs_w = w * img_width
        abs_h = h * img_height
        return {
            "class_id": class_id,
            "type": "bbox",
            "bbox": [abs_x, abs_y, abs_w, abs_h],
            "area": float(abs_w * abs_h),
        }

    if len(coords) >= 6 and len(coords) % 2 == 0:
        points = [
            (coords[i] * img_width, coords[i + 1] * img_height)
            for i in range(0, len(coords), 2)
        ]
        area = polygon_area(points)
        bbox = recompute_bbox_from_polygon(points)
        return {
            "class_id": class_id,
            "type": "polygon",
            "polygon": points,
            "bbox": bbox,
            "area": float(area),
        }

    return None


def recompute_bbox_from_polygon(
    points: List[Tuple[float, float]],
) -> List[float]:
    """Compute ``[x, y, w, h]`` bounding box from polygon points."""
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Polygon area via the shoelace formula."""
    if len(points) < 3:
        return 0.0
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0
