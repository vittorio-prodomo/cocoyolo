"""cocoyolo - Bidirectional COCO ↔ YOLO format converter.

Handles the tricky edge cases: masks with holes and disjoint regions.

Usage::

    from cocoyolo import coco_to_yolo, yolo_to_coco

    # COCO → YOLO (with hole/disjoint strategies)
    coco_to_yolo("path/to/coco", "path/to/yolo")
    coco_to_yolo("path/to/coco", "path/to/yolo",
                 hole_strategy="fill", disjoint_strategy="split")

    # YOLO → COCO
    yolo_to_coco("path/to/yolo", "path/to/coco")
"""

from .coco_to_yolo import convert as coco_to_yolo
from .dataset import COCODatasetInfo, COCOSplit, load_coco_dataset
from .dataset_yolo import YOLODatasetInfo, YOLOSplit, load_yolo_dataset
from .geometry import mask_to_polygons
from .io_utils import decode_rle
from .yolo_to_coco import convert_yolo_to_coco as yolo_to_coco

__all__ = [
    "coco_to_yolo",
    "yolo_to_coco",
    "load_coco_dataset",
    "load_yolo_dataset",
    "mask_to_polygons",
    "COCODatasetInfo",
    "COCOSplit",
    "YOLODatasetInfo",
    "YOLOSplit",
    "decode_rle",
]

__version__ = "0.1.0"
