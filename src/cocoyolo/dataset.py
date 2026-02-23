"""COCO dataset detection and loading.

Handles the three common COCO directory layouts:

    COCO-A:  images/{split}/ + annotations/instances_{split}.json
    COCO-B:  {split}/_annotations.coco.json   (Roboflow inline)
    COCO-C:  annotations/*.json + images/     (flat, single-split)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .io_utils import find_coco_image

logger = logging.getLogger(__name__)

_KNOWN_SPLIT_DIRS = {"train", "val", "valid", "test"}
_SPLIT_ALIASES = {"valid": "val"}


def _normalise_split(name: str) -> str:
    return _SPLIT_ALIASES.get(name.lower(), name.lower())


@dataclass
class COCOSplit:
    """One split of a COCO dataset."""

    name: str
    ann_path: Path
    images_root: Optional[Path] = None
    image_paths: List[Path] = field(default_factory=list)


@dataclass
class COCODatasetInfo:
    """Metadata for a detected COCO dataset."""

    root_dir: Path
    splits: Dict[str, COCOSplit] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    total_images: int = 0

    @property
    def split_names(self) -> List[str]:
        return list(self.splits.keys())


def load_coco_dataset(
    dataset_dir: Union[str, Path], verbose: bool = True
) -> COCODatasetInfo:
    """Auto-detect COCO layout and return dataset metadata.

    Tries COCO-A, COCO-B, then COCO-C in that order.

    Args:
        dataset_dir: Root directory of the COCO dataset.
        verbose: Log progress.

    Returns:
        A :class:`COCODatasetInfo` describing the dataset.

    Raises:
        FileNotFoundError: If no COCO annotations are found.
    """
    root = Path(dataset_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    splits = _try_coco_a(root) or _try_coco_b(root) or _try_coco_c(root)
    if not splits:
        raise FileNotFoundError(
            f"No COCO annotations found in any recognised layout under {root}"
        )

    class_names = _load_class_names(splits)
    total = sum(len(s.image_paths) for s in splits.values())

    if verbose:
        logger.info(
            "Loaded COCO dataset: %d splits, %d classes, %d images",
            len(splits), len(class_names), total,
        )

    return COCODatasetInfo(
        root_dir=root,
        splits=splits,
        class_names=class_names,
        total_images=total,
    )


# ------------------------------------------------------------------
# Variant detectors
# ------------------------------------------------------------------


def _try_coco_a(root: Path) -> Optional[Dict[str, COCOSplit]]:
    """Standard COCO: ``annotations/instances_{split}.json``."""
    ann_dir = root / "annotations"
    if not ann_dir.is_dir():
        return None

    splits: Dict[str, COCOSplit] = {}
    for json_path in sorted(ann_dir.glob("*.json")):
        split_name = _split_name_from_json(json_path.name)
        if split_name is None:
            continue
        image_paths = _resolve_image_paths(json_path, root)
        splits[split_name] = COCOSplit(
            name=split_name,
            ann_path=json_path,
            image_paths=image_paths,
        )

    return splits or None


def _try_coco_b(root: Path) -> Optional[Dict[str, COCOSplit]]:
    """Roboflow inline: ``{split}/_annotations.coco.json``."""
    splits: Dict[str, COCOSplit] = {}
    for dir_name in ("train", "valid", "val", "test"):
        json_path = root / dir_name / "_annotations.coco.json"
        if not json_path.exists():
            continue
        split_name = _normalise_split(dir_name)
        image_paths = _resolve_image_paths(json_path, json_path.parent)
        splits[split_name] = COCOSplit(
            name=split_name,
            ann_path=json_path,
            images_root=json_path.parent,
            image_paths=image_paths,
        )
    return splits or None


def _try_coco_c(root: Path) -> Optional[Dict[str, COCOSplit]]:
    """Flat COCO: a single JSON in ``annotations/`` with no split dirs."""
    ann_dir = root / "annotations"
    if not ann_dir.is_dir():
        return None
    jsons = sorted(ann_dir.glob("*.json"))
    if not jsons:
        return None
    json_path = jsons[0]
    image_paths = _resolve_image_paths(json_path, root)
    return {
        "train": COCOSplit(
            name="train",
            ann_path=json_path,
            image_paths=image_paths,
        )
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _split_name_from_json(filename: str) -> Optional[str]:
    """Extract a split name from a COCO JSON filename."""
    lower = filename.lower()
    for suffix, split in [
        ("_train.json", "train"),
        ("_train2017.json", "train"),
        ("_val.json", "val"),
        ("_val2017.json", "val"),
        ("_validation.json", "val"),
        ("_test.json", "test"),
        ("_test2017.json", "test"),
        ("_default.json", "train"),
    ]:
        if lower.endswith(suffix):
            return split
    return None


def _resolve_image_paths(json_path: Path, search_root: Path) -> List[Path]:
    """Read image filenames from a COCO JSON and resolve them on disk."""
    with open(json_path, "r") as fh:
        data = json.load(fh)

    paths: List[Path] = []
    for img_entry in data.get("images", []):
        fn = img_entry.get("file_name", "")
        if not fn:
            continue
        direct = search_root / fn
        if direct.exists():
            paths.append(direct)
            continue
        via_images = search_root / "images" / fn
        if via_images.exists():
            paths.append(via_images)
            continue
        found = find_coco_image(Path(fn).name, search_root)
        if found:
            paths.append(found)
    return paths


def _load_class_names(splits: Dict[str, COCOSplit]) -> Dict[int, str]:
    """Load category names from the first available COCO JSON."""
    for split in splits.values():
        if split.ann_path and split.ann_path.exists():
            with open(split.ann_path, "r") as fh:
                data = json.load(fh)
            return {
                int(cat["id"]): str(cat["name"])
                for cat in data.get("categories", [])
            }
    return {}
