"""COCO to YOLO format converter.

Supports bounding-box and instance-segmentation annotations, including
RLE-encoded masks, multi-contour (disjoint) shapes, and polygons with holes.

YOLO requires uniform label types within a dataset: either all bounding
boxes (detection) or all polygons (segmentation).  This converter
auto-detects the task type from the COCO annotations and enforces
consistency.  If the dataset is mixed, the user must explicitly choose
``task="detect"`` to output only bounding boxes.
"""

import json
import logging
import multiprocessing
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from .dataset import COCODatasetInfo, COCOSplit, load_coco_dataset
from .geometry import bridge_disjoint, group_contours, mask_to_polygons
from .io_utils import build_image_index, create_data_yaml, decode_rle

logger = logging.getLogger(__name__)

VALID_HOLE_STRATEGIES = ("fill", "bridge")
VALID_DISJOINT_STRATEGIES = ("split", "bridge")
VALID_TASKS = ("auto", "detect", "segment")


# ------------------------------------------------------------------
# Annotation type detection
# ------------------------------------------------------------------


def _ann_has_segmentation(ann: Dict) -> bool:
    """Return True if a COCO annotation has meaningful segmentation data."""
    seg = ann.get("segmentation")
    if not seg:
        return False
    if isinstance(seg, dict):
        return bool(seg)  # RLE
    if isinstance(seg, list):
        return any(
            (isinstance(p, list) and len(p) >= 6) for p in seg
        )
    return False


def _detect_task_type(
    annotations: List[Dict],
) -> Tuple[str, int, int]:
    """Scan annotations and determine the dataset's task type.

    Returns:
        ``(task, n_with_seg, n_bbox_only)``
        where *task* is ``"detect"`` or ``"segment"``.

    Raises:
        ValueError: If the dataset is mixed (some annotations have
            segmentation, some don't) and no explicit task was chosen.
    """
    n_with_seg = 0
    n_bbox_only = 0

    for ann in annotations:
        if _ann_has_segmentation(ann):
            n_with_seg += 1
        elif "bbox" in ann:
            n_bbox_only += 1

    if n_with_seg == 0:
        return "detect", n_with_seg, n_bbox_only
    if n_bbox_only == 0:
        return "segment", n_with_seg, n_bbox_only

    # Mixed — caller must decide
    raise ValueError(
        f"Mixed annotation types: {n_with_seg} annotations have "
        f"segmentation masks, {n_bbox_only} have only bounding boxes.\n"
        f"YOLO requires uniform label types within a dataset.\n"
        f"Use --task detect  to convert everything to bounding boxes.\n"
        f"Use --task segment to keep only the {n_with_seg} annotations "
        f"that have segmentation (the {n_bbox_only} bbox-only annotations "
        f"will be skipped)."
    )


# ------------------------------------------------------------------
# Conversion statistics
# ------------------------------------------------------------------


@dataclass
class ConversionStats:
    """Tracks what happened during COCO → YOLO conversion."""

    # Task type resolved for this conversion
    task: str = ""

    # Annotation types
    bbox_only: int = 0
    polygon_single: int = 0
    polygon_disjoint: int = 0
    rle_simple: int = 0
    rle_with_holes: int = 0
    rle_disjoint: int = 0

    # Strategy applications
    holes_bridged: int = 0
    holes_filled: int = 0
    disjoint_bridged: int = 0
    disjoint_split: int = 0

    # Skips
    images_not_found: int = 0
    rle_decode_failures: int = 0
    skipped_unknown_category: int = 0
    skipped_no_segmentation: int = 0

    # Per-split tracking
    split_stats: Dict[str, "ConversionStats"] = field(default_factory=dict)

    @property
    def total_annotations(self) -> int:
        return (
            self.bbox_only + self.polygon_single + self.polygon_disjoint
            + self.rle_simple + self.rle_with_holes + self.rle_disjoint
        )

    @property
    def total_segmentation(self) -> int:
        return (
            self.polygon_single + self.polygon_disjoint
            + self.rle_simple + self.rle_with_holes + self.rle_disjoint
        )

    @property
    def total_edge_cases(self) -> int:
        return (
            self.polygon_disjoint + self.rle_with_holes + self.rle_disjoint
        )

    def merge(self, other: "ConversionStats") -> None:
        self.bbox_only += other.bbox_only
        self.polygon_single += other.polygon_single
        self.polygon_disjoint += other.polygon_disjoint
        self.rle_simple += other.rle_simple
        self.rle_with_holes += other.rle_with_holes
        self.rle_disjoint += other.rle_disjoint
        self.holes_bridged += other.holes_bridged
        self.holes_filled += other.holes_filled
        self.disjoint_bridged += other.disjoint_bridged
        self.disjoint_split += other.disjoint_split
        self.images_not_found += other.images_not_found
        self.rle_decode_failures += other.rle_decode_failures
        self.skipped_unknown_category += other.skipped_unknown_category
        self.skipped_no_segmentation += other.skipped_no_segmentation

    def format_summary(
        self, hole_strategy: str, disjoint_strategy: str
    ) -> str:
        lines = []
        lines.append(f"  YOLO task type:        {self.task}")
        lines.append(f"  Annotations processed: {self.total_annotations}")
        if self.task == "detect":
            lines.append(f"    Bbox:                {self.bbox_only}")
        else:
            lines.append(f"    Bbox-only:           {self.bbox_only}")
            lines.append(f"    Polygon (single):    {self.polygon_single}")
            lines.append(f"    Polygon (disjoint):  {self.polygon_disjoint}")
            lines.append(f"    RLE (simple):        {self.rle_simple}")
            lines.append(f"    RLE (with holes):    {self.rle_with_holes}")
            lines.append(f"    RLE (disjoint):      {self.rle_disjoint}")

            if self.total_edge_cases > 0:
                lines.append(f"  Edge cases: {self.total_edge_cases}")
                if self.rle_with_holes > 0:
                    action = "bridged" if hole_strategy == "bridge" else "filled"
                    count = self.holes_bridged if hole_strategy == "bridge" else self.holes_filled
                    lines.append(
                        f"    Holes ({hole_strategy}):    "
                        f"{count} contour(s) {action}"
                    )
                if self.polygon_disjoint + self.rle_disjoint > 0:
                    action = "bridged" if disjoint_strategy == "bridge" else "split"
                    count = self.disjoint_bridged if disjoint_strategy == "bridge" else self.disjoint_split
                    lines.append(
                        f"    Disjoint ({disjoint_strategy}): "
                        f"{count} annotation(s) {action}"
                    )

        warnings = []
        if self.images_not_found:
            warnings.append(f"    Images not found:    {self.images_not_found}")
        if self.rle_decode_failures:
            warnings.append(f"    RLE decode failures: {self.rle_decode_failures}")
        if self.skipped_unknown_category:
            warnings.append(f"    Unknown categories:  {self.skipped_unknown_category}")
        if self.skipped_no_segmentation:
            warnings.append(
                f"    Skipped (no seg):    {self.skipped_no_segmentation}"
            )
        if warnings:
            lines.append("  Warnings:")
            lines.extend(warnings)

        return "\n".join(lines)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def convert(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    contour_approx_factor: float = 0.001,
    hole_strategy: str = "bridge",
    disjoint_strategy: str = "bridge",
    task: str = "auto",
    image_mode: str = "copy",
    workers: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[COCODatasetInfo, ConversionStats]:
    """Convert a COCO dataset to YOLO Ultralytics format.

    Args:
        input_path: Root of the source COCO dataset.
        output_path: Where to write the YOLO dataset.
        contour_approx_factor: ``cv2.approxPolyDP`` epsilon as a fraction
            of the contour arc length.
        hole_strategy: How to handle holes in RLE masks.
            ``"fill"`` ignores holes; ``"bridge"`` preserves them via
            inverse bridges.
        disjoint_strategy: How to handle disjoint regions.
            ``"split"`` writes each region as a separate annotation;
            ``"bridge"`` connects them via nearest-neighbour chain
            with zero-width bridges.
        task: YOLO output task type.
            ``"auto"`` detects from annotations (errors on mixed).
            ``"detect"`` forces bounding-box output.
            ``"segment"`` forces polygon output.
        image_mode: How to transfer images to the output directory.
            ``"copy"`` (default) makes full copies.
            ``"symlink"`` creates relative symbolic links.
            ``"hardlink"`` creates hard links.
        workers: Number of parallel worker processes.  ``None`` (default)
            uses all available CPU cores.  ``1`` disables
            multiprocessing.
        verbose: Show progress bars.

    Returns:
        A tuple of ``(COCODatasetInfo, ConversionStats)``.

    Raises:
        ValueError: If an invalid strategy or task is given, or if the
            dataset has mixed annotation types in ``"auto"`` mode.
        FileNotFoundError: If the input dataset is not found.
    """
    if hole_strategy not in VALID_HOLE_STRATEGIES:
        raise ValueError(
            f"hole_strategy must be one of {VALID_HOLE_STRATEGIES}, "
            f"got {hole_strategy!r}"
        )
    if disjoint_strategy not in VALID_DISJOINT_STRATEGIES:
        raise ValueError(
            f"disjoint_strategy must be one of {VALID_DISJOINT_STRATEGIES}, "
            f"got {disjoint_strategy!r}"
        )
    if task not in VALID_TASKS:
        raise ValueError(
            f"task must be one of {VALID_TASKS}, got {task!r}"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)

    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    logger.info("COCO -> YOLO: %s -> %s", input_path, output_path)
    logger.info(
        "Strategies: holes=%s, disjoint=%s, workers=%d",
        hole_strategy, disjoint_strategy, workers,
    )

    src_info = load_coco_dataset(input_path, verbose=verbose)
    total_stats = ConversionStats()

    # Create output directories
    for split in src_info.splits:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_class_names: Dict[int, str] = {}
    for split_name, split in src_info.splits.items():
        logger.info("Converting split: %s", split_name)
        split_classes, split_stats = _convert_split(
            split, src_info, output_path,
            contour_approx_factor, hole_strategy, disjoint_strategy,
            task, image_mode, workers, verbose,
        )
        all_class_names.update(split_classes)
        total_stats.split_stats[split_name] = split_stats
        total_stats.merge(split_stats)

    # Propagate resolved task from first split
    if not total_stats.task:
        for ss in total_stats.split_stats.values():
            if ss.task:
                total_stats.task = ss.task
                break

    # Build sequential 0-based class mapping for the YAML
    ordered = sorted(set(all_class_names.values()))
    yolo_classes = {i: name for i, name in enumerate(ordered)}
    create_data_yaml(output_path, yolo_classes, list(src_info.splits.keys()))

    logger.info("COCO -> YOLO conversion complete")
    return src_info, total_stats


# ------------------------------------------------------------------
# Image index + file transfer helpers
# ------------------------------------------------------------------


_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def _transfer_file(src: Path, dst: Path, mode: str) -> None:
    """Copy, symlink, or hardlink *src* to *dst*.

    Falls back to copy if the requested link mode fails.
    Silently skips the transfer when *src* and *dst* resolve to the
    same file (in-place conversion).
    """
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        try:
            rel = os.path.relpath(src, dst.parent)
            os.symlink(rel, dst)
            return
        except (OSError, ValueError):
            pass
    elif mode == "hardlink":
        try:
            os.link(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


# ------------------------------------------------------------------
# Per-image worker (module-level for multiprocessing picklability)
# ------------------------------------------------------------------


def _process_image(work_item: tuple) -> ConversionStats:
    """Process one image: transfer file and write YOLO labels.

    Args:
        work_item: Tuple of
            ``(src_img, dst_img, dst_lbl, img_w, img_h, annotations,
            cat_to_yolo, task, approx_factor, hole_strategy,
            disjoint_strategy, image_mode)``.

    Returns:
        ``ConversionStats`` for this single image.
    """
    (
        src_img, dst_img, dst_lbl,
        img_w, img_h, annotations, cat_to_yolo,
        task, approx_factor, hole_strategy, disjoint_strategy, image_mode,
    ) = work_item

    stats = ConversionStats()

    if src_img is None:
        stats.images_not_found += 1
        return stats

    _transfer_file(src_img, dst_img, image_mode)

    if task == "detect":
        _write_yolo_detect_labels(
            annotations, dst_lbl, cat_to_yolo, img_w, img_h, stats,
        )
    else:
        _write_yolo_segment_labels(
            annotations, dst_lbl, cat_to_yolo, img_w, img_h,
            approx_factor, hole_strategy, disjoint_strategy, stats,
        )

    return stats


# ------------------------------------------------------------------
# Split conversion
# ------------------------------------------------------------------


def _convert_split(
    split: COCOSplit,
    info: COCODatasetInfo,
    output: Path,
    approx_factor: float,
    hole_strategy: str,
    disjoint_strategy: str,
    task: str,
    image_mode: str,
    workers: int,
    verbose: bool,
) -> Tuple[Dict[int, str], ConversionStats]:
    """Convert one COCO split -> YOLO.  Returns (category_map, stats)."""
    stats = ConversionStats()

    with open(split.ann_path, "r") as fh:
        coco = json.load(fh)

    img_map = {img["id"]: img for img in coco.get("images", [])}
    cats = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    all_anns = coco.get("annotations", [])

    # ---- Resolve task type for this split ----
    if task == "auto":
        resolved, n_seg, n_bbox = _detect_task_type(all_anns)
    elif task == "detect":
        resolved = "detect"
        n_seg = sum(1 for a in all_anns if _ann_has_segmentation(a))
        n_bbox = len(all_anns) - n_seg
    else:
        resolved = "segment"
        n_seg = sum(1 for a in all_anns if _ann_has_segmentation(a))
        n_bbox = len(all_anns) - n_seg

    stats.task = resolved
    logger.info(
        "  Split %s: task=%s (%d with segmentation, %d bbox-only)",
        split.name, resolved, n_seg, n_bbox,
    )

    # Build COCO-id -> sequential 0-based YOLO-id mapping
    cat_to_yolo: Dict[int, int] = {}
    ordered_names = sorted(set(cats.values()))
    name_to_yolo = {name: idx for idx, name in enumerate(ordered_names)}
    for coco_id, name in cats.items():
        cat_to_yolo[coco_id] = name_to_yolo[name]

    anns_by_img: Dict[int, list] = defaultdict(list)
    for ann in all_anns:
        anns_by_img[ann["image_id"]].append(ann)

    # Pre-resolve all image paths (single directory scan, with duplicate check)
    search_root = split.images_root or info.root_dir
    image_index = build_image_index(search_root, _IMAGE_EXTENSIONS)

    # Build work items
    work_items = []
    for img_id, img_info in img_map.items():
        filename = img_info["file_name"]
        src_img = image_index.get(Path(filename).name)
        dst_img = output / "images" / split.name / Path(filename).name
        dst_lbl = output / "labels" / split.name / (Path(filename).stem + ".txt")

        work_items.append((
            src_img, dst_img, dst_lbl,
            img_info["width"], img_info["height"],
            anns_by_img[img_id], cat_to_yolo,
            resolved, approx_factor,
            hole_strategy, disjoint_strategy, image_mode,
        ))

    # Dispatch — sequential or parallel
    effective_workers = min(workers, len(work_items)) if work_items else 1
    if effective_workers <= 1:
        for item in tqdm(
            work_items, desc=f"  {split.name}", disable=not verbose,
        ):
            stats.merge(_process_image(item))
    else:
        with multiprocessing.Pool(effective_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_image, work_items),
                total=len(work_items),
                desc=f"  {split.name}",
                disable=not verbose,
            ):
                stats.merge(result)

    return cats, stats


# ------------------------------------------------------------------
# Label writing — detection (bbox only)
# ------------------------------------------------------------------


def _write_yolo_detect_labels(
    annotations: List[Dict],
    output_file: Path,
    cat_map: Dict[int, int],
    img_w: int,
    img_h: int,
    stats: ConversionStats,
) -> None:
    """Write YOLO detection labels: ``class xc yc w h`` (normalised).

    In detection mode, every annotation is converted to a bounding box.
    If the annotation has segmentation but no explicit bbox, we compute
    the bbox from the segmentation polygon.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fh:
        for ann in annotations:
            cid = ann.get("category_id")
            if cid not in cat_map:
                stats.skipped_unknown_category += 1
                continue
            yolo_id = cat_map[cid]

            if "bbox" not in ann:
                continue

            stats.bbox_only += 1
            x, y, w, h = ann["bbox"]
            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            fh.write(
                f"{yolo_id} {xc:.6f} {yc:.6f} "
                f"{w / img_w:.6f} {h / img_h:.6f}\n"
            )


# ------------------------------------------------------------------
# Label writing — segmentation (polygons)
# ------------------------------------------------------------------


def _write_yolo_segment_labels(
    annotations: List[Dict],
    output_file: Path,
    cat_map: Dict[int, int],
    img_w: int,
    img_h: int,
    approx_factor: float,
    hole_strategy: str,
    disjoint_strategy: str,
    stats: ConversionStats,
) -> None:
    """Write YOLO segmentation labels: ``class x1 y1 x2 y2 ...`` (normalised).

    In segmentation mode, annotations without segmentation data are skipped.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as fh:
        for ann in annotations:
            cid = ann.get("category_id")
            if cid not in cat_map:
                stats.skipped_unknown_category += 1
                continue
            yolo_id = cat_map[cid]

            if not _ann_has_segmentation(ann):
                stats.skipped_no_segmentation += 1
                continue

            seg = ann["segmentation"]
            polys = _segmentation_to_yolo(
                seg, img_w, img_h, approx_factor,
                hole_strategy, disjoint_strategy, stats,
            )
            for coords in polys:
                if coords:
                    line = " ".join(f"{c:.6f}" for c in coords)
                    fh.write(f"{yolo_id} {line}\n")


# ------------------------------------------------------------------
# Segmentation conversion
# ------------------------------------------------------------------


def _segmentation_to_yolo(
    seg: Union[Dict, List],
    img_w: int,
    img_h: int,
    approx_factor: float,
    hole_strategy: str,
    disjoint_strategy: str,
    stats: ConversionStats,
) -> List[List[float]]:
    """Convert a COCO segmentation field to YOLO normalised polygons."""
    if isinstance(seg, dict):
        return _rle_to_yolo(
            seg, img_w, img_h, approx_factor,
            hole_strategy, disjoint_strategy, stats,
        )
    if isinstance(seg, list):
        return _polygon_list_to_yolo(
            seg, img_w, img_h, disjoint_strategy, stats,
        )
    return []


def _polygon_list_to_yolo(
    seg_list: List[List[float]],
    img_w: int,
    img_h: int,
    disjoint_strategy: str,
    stats: ConversionStats,
) -> List[List[float]]:
    """Convert COCO polygon-list segmentation to normalised YOLO coords.

    COCO polygon lists cannot represent holes (all sub-polygons are
    additive outer regions), so only *disjoint_strategy* applies.
    """
    valid = [p for p in seg_list if len(p) >= 6]
    if not valid:
        return []

    if len(valid) == 1:
        stats.polygon_single += 1
        return [_normalise_flat_polygon(valid[0], img_w, img_h)]

    # Multiple sub-polygons → disjoint
    stats.polygon_disjoint += 1

    if disjoint_strategy == "split":
        stats.disjoint_split += 1
        return [_normalise_flat_polygon(p, img_w, img_h) for p in valid]

    # Bridge
    stats.disjoint_bridged += 1
    point_lists = [_flat_to_points(p) for p in valid]
    bridged = bridge_disjoint(point_lists)
    return [_normalize_point_list(bridged, img_w, img_h)]


def _rle_to_yolo(
    rle: Dict,
    img_w: int,
    img_h: int,
    approx_factor: float,
    hole_strategy: str,
    disjoint_strategy: str,
    stats: ConversionStats,
) -> List[List[float]]:
    """Decode RLE -> contours -> normalised YOLO polygon coordinates.

    Delegates the core mask-to-polygon conversion to
    :func:`~cocoyolo.geometry.mask_to_polygons`, then normalises the
    pixel-space result and updates conversion statistics.
    """
    try:
        mask = decode_rle(rle)
    except Exception as exc:
        logger.warning("Failed to decode RLE: %s", exc)
        stats.rle_decode_failures += 1
        return []

    polys = mask_to_polygons(
        mask,
        approx_factor=approx_factor,
        hole_strategy=hole_strategy,
        disjoint_strategy=disjoint_strategy,
    )
    if not polys:
        return []

    # Update statistics (mask_to_polygons doesn't track these)
    # Inspect the mask to determine hole/disjoint classification
    contours, hierarchy = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255,
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    groups = group_contours(contours, hierarchy)
    has_holes = any(len(holes) > 0 for _, holes in groups)
    is_disjoint = len(groups) > 1

    if has_holes:
        stats.rle_with_holes += 1
        if hole_strategy == "fill":
            stats.holes_filled += sum(1 for _, h in groups if h)
        else:
            stats.holes_bridged += sum(1 for _, h in groups if h)
    elif is_disjoint:
        stats.rle_disjoint += 1
        if disjoint_strategy == "split":
            stats.disjoint_split += 1
        else:
            stats.disjoint_bridged += 1
    else:
        stats.rle_simple += 1

    return [_normalize_point_list(p, img_w, img_h) for p in polys]


# ------------------------------------------------------------------
# Coordinate helpers
# ------------------------------------------------------------------


def _normalize_point_list(
    pts: List[List[float]], img_w: int, img_h: int
) -> List[float]:
    """Flatten ``[[x, y], ...]`` to ``[x/w, y/h, ...]`` normalised coords."""
    coords: List[float] = []
    for x, y in pts:
        coords.extend([x / img_w, y / img_h])
    return coords


def _normalise_flat_polygon(
    flat: List[float], img_w: int, img_h: int
) -> List[float]:
    """Normalise a flat ``[x1, y1, x2, y2, ...]`` COCO polygon."""
    return [
        flat[i] / img_w if i % 2 == 0 else flat[i] / img_h
        for i in range(len(flat))
    ]


def _flat_to_points(flat: List[float]) -> List[List[float]]:
    """Convert ``[x1, y1, x2, y2, ...]`` to ``[[x1, y1], [x2, y2], ...]``."""
    return [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]
