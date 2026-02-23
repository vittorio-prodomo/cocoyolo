"""YOLO dataset detection and loading.

Handles the two common YOLO directory layouts:

    YOLO-A:  images/{split}/ + labels/{split}/ + data.yaml
    YOLO-B:  {split}/images/ + {split}/labels/ + data.yaml
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)

_SPLIT_ALIASES = {"valid": "val"}
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def _normalise_split(name: str) -> str:
    return _SPLIT_ALIASES.get(name.lower(), name.lower())


@dataclass
class YOLOSplit:
    """One split of a YOLO dataset."""

    name: str
    image_paths: List[Path] = field(default_factory=list)
    label_paths: List[Path] = field(default_factory=list)


@dataclass
class YOLODatasetInfo:
    """Metadata for a detected YOLO dataset."""

    root_dir: Path
    splits: Dict[str, YOLOSplit] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    total_images: int = 0

    @property
    def split_names(self) -> List[str]:
        return list(self.splits.keys())


def load_yolo_dataset(
    dataset_dir: Union[str, Path], verbose: bool = True
) -> YOLODatasetInfo:
    """Auto-detect YOLO layout and return dataset metadata.

    Tries YOLO-A then YOLO-B.

    Args:
        dataset_dir: Root directory of the YOLO dataset.
        verbose: Log progress.

    Returns:
        A :class:`YOLODatasetInfo` describing the dataset.

    Raises:
        FileNotFoundError: If no YOLO dataset structure is found.
    """
    root = Path(dataset_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_names = _parse_data_yaml(root)

    splits = _try_yolo_a(root) or _try_yolo_b(root)
    if not splits:
        raise FileNotFoundError(
            f"No YOLO dataset found in any recognised layout under {root}. "
            f"Expected images/ + labels/ directories."
        )

    total = sum(len(s.image_paths) for s in splits.values())

    if verbose:
        logger.info(
            "Loaded YOLO dataset: %d splits, %d classes, %d images",
            len(splits), len(class_names), total,
        )

    return YOLODatasetInfo(
        root_dir=root,
        splits=splits,
        class_names=class_names,
        total_images=total,
    )


# ------------------------------------------------------------------
# Variant detectors
# ------------------------------------------------------------------


def _try_yolo_a(root: Path) -> Optional[Dict[str, YOLOSplit]]:
    """YOLO-A: ``images/{split}/`` + ``labels/{split}/``."""
    images_dir = root / "images"
    labels_dir = root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return None

    splits: Dict[str, YOLOSplit] = {}
    for sub in sorted(images_dir.iterdir()):
        if not sub.is_dir():
            continue
        split_name = _normalise_split(sub.name)
        label_sub = labels_dir / sub.name
        if not label_sub.is_dir():
            label_sub = labels_dir / split_name

        image_paths = _find_images(sub)
        label_paths = _match_labels(image_paths, label_sub)

        splits[split_name] = YOLOSplit(
            name=split_name,
            image_paths=image_paths,
            label_paths=label_paths,
        )

    return splits or None


def _try_yolo_b(root: Path) -> Optional[Dict[str, YOLOSplit]]:
    """YOLO-B: ``{split}/images/`` + ``{split}/labels/``."""
    splits: Dict[str, YOLOSplit] = {}
    for dir_name in ("train", "valid", "val", "test"):
        split_dir = root / dir_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.is_dir():
            continue
        split_name = _normalise_split(dir_name)

        image_paths = _find_images(img_dir)
        label_paths = _match_labels(image_paths, lbl_dir)

        splits[split_name] = YOLOSplit(
            name=split_name,
            image_paths=image_paths,
            label_paths=label_paths,
        )

    return splits or None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_data_yaml(root: Path) -> Dict[int, str]:
    """Read class names from ``data.yaml``."""
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        logger.warning("No data.yaml found at %s", root)
        return {}

    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def _find_images(directory: Path) -> List[Path]:
    """Return sorted list of image files in *directory*."""
    images: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def _match_labels(
    image_paths: List[Path], label_dir: Path
) -> List[Path]:
    """Match each image to its corresponding ``.txt`` label file."""
    labels: List[Path] = []
    if not label_dir.is_dir():
        return labels
    for img in image_paths:
        lbl = label_dir / (img.stem + ".txt")
        if lbl.exists():
            labels.append(lbl)
    return labels
