"""COCO dataset detection and loading.

Auto-discovers COCO annotation JSONs and resolves images by basename
via a single recursive scan of the dataset root.  No rigid directory
structure is assumed for images — they can live at any depth.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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
    """Auto-detect COCO annotation files and return dataset metadata.

    Discovers COCO JSON files by scanning (in order):

    1. ``annotations/`` directory for ``*.json``
    2. Known split directories (``train/``, ``val/``, ``test/``) for
       inline ``_annotations.coco.json``
    3. Root-level ``*.json`` files

    Each candidate is validated as a COCO file (must contain both
    ``"images"`` and ``"annotations"`` keys).  Images are not resolved
    here — they are looked up by basename during conversion via a
    single recursive scan of the dataset root.

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

    splits = _discover_coco_jsons(root)
    if not splits:
        raise FileNotFoundError(
            f"No COCO annotations found under {root}"
        )

    class_names = _load_class_names(splits)
    total = _count_json_images(splits)

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
# JSON discovery
# ------------------------------------------------------------------


def _discover_coco_jsons(root: Path) -> Optional[Dict[str, COCOSplit]]:
    """Find all COCO annotation JSONs under *root*.

    Scans three locations in order:

    1. ``annotations/`` directory
    2. Known split directories (``train/``, ``val/``, ``test/``) for
       inline ``_annotations.coco.json``
    3. Root-level ``*.json`` files

    Each candidate is validated and the first valid JSON per split
    name wins.
    """
    candidates: List[Path] = []

    # 1. annotations/ directory (standard COCO)
    ann_dir = root / "annotations"
    if ann_dir.is_dir():
        candidates.extend(sorted(ann_dir.glob("*.json")))

    # 2. Inline JSONs in known split directories (Roboflow)
    for dir_name in sorted(_KNOWN_SPLIT_DIRS):
        json_path = root / dir_name / "_annotations.coco.json"
        if json_path.exists():
            candidates.append(json_path)

    # 3. Root-level JSON files
    candidates.extend(sorted(root.glob("*.json")))

    # Deduplicate preserving discovery order
    seen: set = set()
    unique: List[Path] = []
    for c in candidates:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(c)

    splits: Dict[str, COCOSplit] = {}
    for json_path in unique:
        if not _is_coco_json(json_path):
            continue
        split_name = _infer_split_name(json_path)
        if split_name in splits:
            continue  # first match per split wins
        splits[split_name] = COCOSplit(
            name=split_name,
            ann_path=json_path,
        )

    return splits or None


def _is_coco_json(path: Path) -> bool:
    """Check whether *path* is a COCO annotation JSON."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return (
            isinstance(data, dict)
            and "images" in data
            and "annotations" in data
        )
    except (json.JSONDecodeError, OSError):
        return False


def _infer_split_name(json_path: Path) -> str:
    """Determine the split name for a COCO JSON file.

    Tries (in order):

    1. Filename pattern (``instances_train.json`` → ``train``).
    2. Parent directory name if it's a known split dir.
    3. Falls back to ``"train"``.
    """
    # From filename
    name = _split_name_from_json(json_path.name)
    if name is not None:
        return name

    # From parent directory
    parent = json_path.parent.name.lower()
    if parent in _KNOWN_SPLIT_DIRS:
        return _normalise_split(parent)

    return "train"


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


def _count_json_images(splits: Dict[str, COCOSplit]) -> int:
    """Count total images across all splits by reading their JSONs."""
    total = 0
    for split in splits.values():
        try:
            with open(split.ann_path, "r") as f:
                data = json.load(f)
            total += len(data.get("images", []))
        except (json.JSONDecodeError, OSError):
            pass
    return total


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
