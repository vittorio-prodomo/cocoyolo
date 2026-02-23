"""YOLO to COCO format converter.

Supports bounding-box (detection) and polygon (instance-segmentation)
annotations.  Each YOLO label line becomes one COCO annotation with a
polygon or bbox, depending on the number of coordinate values.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

from tqdm import tqdm

from .annotations import parse_yolo_line, recompute_bbox_from_polygon
from .dataset_yolo import YOLODatasetInfo, YOLOSplit, load_yolo_dataset
from .image_utils import get_image_size
from .io_utils import FileLinker

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Conversion statistics
# ------------------------------------------------------------------


@dataclass
class ConversionStats:
    """Tracks what happened during YOLO → COCO conversion."""

    bbox_only: int = 0
    polygon: int = 0
    images_skipped: int = 0
    malformed_lines: int = 0
    skipped_unknown_category: int = 0
    skipped_small_bbox: int = 0

    split_stats: Dict[str, "ConversionStats"] = field(default_factory=dict)

    @property
    def total_annotations(self) -> int:
        return self.bbox_only + self.polygon

    def merge(self, other: "ConversionStats") -> None:
        self.bbox_only += other.bbox_only
        self.polygon += other.polygon
        self.images_skipped += other.images_skipped
        self.malformed_lines += other.malformed_lines
        self.skipped_unknown_category += other.skipped_unknown_category
        self.skipped_small_bbox += other.skipped_small_bbox

    def format_summary(self) -> str:
        lines = []
        lines.append(f"  Annotations converted: {self.total_annotations}")
        lines.append(f"    Bbox-only:           {self.bbox_only}")
        lines.append(f"    Polygon:             {self.polygon}")
        if self.images_skipped:
            lines.append(f"  Warnings:")
            lines.append(f"    Images skipped:      {self.images_skipped}")
        if self.malformed_lines:
            lines.append(f"    Malformed lines:     {self.malformed_lines}")
        if self.skipped_unknown_category:
            lines.append(f"    Unknown categories:  {self.skipped_unknown_category}")
        if self.skipped_small_bbox:
            lines.append(f"    Small bboxes dropped:{self.skipped_small_bbox}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def convert_yolo_to_coco(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    keep_zero_indexing: bool = False,
    min_bbox_size: float = 0.0,
    image_mode: str = "copy",
    verbose: bool = True,
) -> Tuple[YOLODatasetInfo, ConversionStats]:
    """Convert a YOLO Ultralytics dataset to COCO format.

    Args:
        input_path: Root of the source YOLO dataset (must contain
            ``data.yaml`` and ``images/``/``labels/`` directories).
        output_path: Where to write the COCO dataset.  Produces the
            standard COCO-A layout: ``images/*.ext`` +
            ``annotations/instances_{split}.json``.
        keep_zero_indexing: If *False* (default), category IDs in the
            output JSON start at 1 (COCO convention).  If *True*,
            they mirror the 0-based YOLO class indices.
        min_bbox_size: Drop bounding boxes whose width **or** height is
            smaller than this value (in pixels).  Defaults to 0 (keep all).
        image_mode: How to transfer images to the output directory.
            ``"copy"`` (default) makes full copies.
            ``"symlink"`` creates relative symbolic links.
            ``"hardlink"`` creates hard links.
        verbose: Show progress bars.

    Returns:
        A tuple of ``(YOLODatasetInfo, ConversionStats)``.

    Raises:
        FileNotFoundError: If the input dataset is not found.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("YOLO -> COCO: %s -> %s", input_path, output_path)

    src_info = load_yolo_dataset(input_path, verbose=verbose)
    linker = FileLinker(mode=image_mode)
    total_stats = ConversionStats()

    try:
        # Create output skeleton (COCO-A: images/{split}/ + annotations/)
        (output_path / "annotations").mkdir(parents=True, exist_ok=True)

        for split_name, split in src_info.splits.items():
            logger.info("Converting split: %s", split_name)
            split_stats = _convert_split(
                split, src_info, output_path, linker,
                keep_zero_indexing, min_bbox_size, verbose,
            )
            total_stats.split_stats[split_name] = split_stats
            total_stats.merge(split_stats)

        logger.info("YOLO -> COCO conversion complete")
        return src_info, total_stats

    except Exception:
        linker.rollback()
        raise


# ------------------------------------------------------------------
# Split conversion
# ------------------------------------------------------------------


def _convert_split(
    split: YOLOSplit,
    info: YOLODatasetInfo,
    output: Path,
    linker: FileLinker,
    keep_zero: bool,
    min_bbox_size: float,
    verbose: bool,
) -> ConversionStats:
    """Convert one YOLO split to a COCO JSON + images directory."""
    stats = ConversionStats()
    coco: Dict = {"images": [], "annotations": [], "categories": []}

    # Categories
    for cid, name in sorted(info.class_names.items()):
        coco_cid = cid if keep_zero else cid + 1
        coco["categories"].append(
            {"id": coco_cid, "name": name, "supercategory": ""}
        )

    # Build stem -> label_path lookup
    label_index: Dict[str, Path] = {lp.stem: lp for lp in split.label_paths}

    image_id = 1
    ann_id = 1

    for img_path in tqdm(
        split.image_paths,
        desc=f"  {split.name}",
        disable=not verbose,
    ):
        try:
            w, h = get_image_size(img_path)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", img_path, exc)
            stats.images_skipped += 1
            continue

        dst_dir = output / "images" / split.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_img = dst_dir / img_path.name
        linker.link(img_path, dst_img)

        coco["images"].append({
            "id": image_id,
            "file_name": f"{split.name}/{img_path.name}",
            "width": w,
            "height": h,
        })

        label_file = label_index.get(img_path.stem)
        if label_file and label_file.exists():
            new_anns = _convert_labels(
                label_file, image_id, info.class_names,
                w, h, keep_zero, min_bbox_size, stats,
            )
            for ann in new_anns:
                ann["id"] = ann_id
                coco["annotations"].append(ann)
                ann_id += 1

        image_id += 1

    out_json = output / "annotations" / f"instances_{split.name}.json"
    with open(out_json, "w") as fh:
        json.dump(coco, fh, indent=2)

    logger.info(
        "Created %s -- %d images, %d annotations",
        out_json.name, len(coco["images"]), len(coco["annotations"]),
    )

    return stats


# ------------------------------------------------------------------
# Label file conversion
# ------------------------------------------------------------------


def _convert_labels(
    label_file: Path,
    image_id: int,
    class_names: Dict[int, str],
    img_w: int,
    img_h: int,
    keep_zero: bool,
    min_bbox_size: float,
    stats: ConversionStats,
) -> List[Dict]:
    """Parse a YOLO ``.txt`` file and return a list of COCO annotation dicts."""
    annotations: List[Dict] = []
    with open(label_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parsed = parse_yolo_line(line, img_w, img_h)
            if parsed is None:
                stats.malformed_lines += 1
                continue
            cid = parsed["class_id"]
            if cid not in class_names:
                stats.skipped_unknown_category += 1
                continue

            bbox = parsed["bbox"]
            if parsed["type"] == "polygon":
                bbox = recompute_bbox_from_polygon(parsed["polygon"])

            # Filter tiny bboxes when requested
            if min_bbox_size > 0 and (bbox[2] < min_bbox_size or bbox[3] < min_bbox_size):
                stats.skipped_small_bbox += 1
                continue

            coco_cid = cid if keep_zero else cid + 1
            ann: Dict = {
                "image_id": image_id,
                "category_id": coco_cid,
                "area": parsed["area"],
                "iscrowd": 0,
            }

            if parsed["type"] == "bbox":
                stats.bbox_only += 1
                ann["bbox"] = bbox
                ann["segmentation"] = []
                annotations.append(ann)

            elif parsed["type"] == "polygon":
                stats.polygon += 1
                points = parsed["polygon"]  # list of (x, y) absolute
                flat = []
                for px, py in points:
                    flat.extend([float(px), float(py)])
                ann["bbox"] = bbox
                ann["segmentation"] = [flat]
                annotations.append(ann)

    return annotations
