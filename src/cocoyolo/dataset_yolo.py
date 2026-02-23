"""YOLO dataset detection and loading.

Discovery is driven entirely by the ``data.yaml`` (or ``dataset.yaml``)
file that every Ultralytics YOLO dataset carries.  This file explicitly
defines the per-split image sub-paths.  Label paths are derived by
replacing the first occurrence of ``images`` with ``labels`` in the
image path — matching the convention used by Ultralytics pipelines.

The input can be either:

- A **directory** that contains a ``data*.yaml`` file, or
- A direct **path to a YAML file** (which may live anywhere).

In the second case the dataset root is determined from the ``path``
entry inside the YAML (resolved relative to the YAML file's own
directory), falling back to the YAML file's parent directory when
``path`` is absent or set to ``"."``.

If no matching YAML file is found the loader aborts with an informative
error, including an example of what a ``data.yaml`` should look like.
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

_DATA_YAML_EXAMPLE = """\
A data.yaml file typically looks like this:

    train: images/train
    val: images/val
    nc: 2
    names:
      0: cat
      1: dog

The 'train' and 'val' keys point to the image directories (relative to
the YAML file).  Label directories are derived automatically by replacing
'images' with 'labels' in each path (e.g. images/train -> labels/train).
"""


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
    """Load a YOLO dataset by reading its ``data.yaml``.

    Args:
        dataset_dir: Either the root directory of the YOLO dataset
            (must contain a ``data*.yaml`` file), or a direct path to
            a YAML file.  When a YAML file is given, the dataset root
            is determined from its ``path`` entry (resolved relative to
            the YAML's own directory), falling back to the YAML's
            parent directory.
        verbose: Log progress.

    Returns:
        A :class:`YOLODatasetInfo` describing the dataset.

    Raises:
        FileNotFoundError: If the path or YAML file is missing.
        ValueError: If the YAML exists but defines no valid splits.
    """
    input_path = Path(dataset_dir).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    # Determine yaml_path and dataset root
    if input_path.is_file():
        yaml_path = input_path
        root = _resolve_dataset_root(yaml_path)
    else:
        root = input_path
        yaml_path = _find_data_yaml(root)
        if yaml_path is None:
            raise FileNotFoundError(
                f"No data.yaml / dataset.yaml found in {root}.\n\n"
                f"Every YOLO Ultralytics dataset must include a YAML file "
                f"(matching the pattern data*.yaml) that declares split "
                f"paths and class names.\n\n"
                + _DATA_YAML_EXAMPLE
            )

    class_names = _parse_class_names(yaml_path)
    splits = _splits_from_yaml(yaml_path, root)

    if not splits:
        raise ValueError(
            f"The YAML file {yaml_path} was found but no valid splits "
            f"could be resolved from it.  Make sure the paths listed "
            f"under 'train', 'val', or 'test' point to existing image "
            f"directories.\n\n"
            + _DATA_YAML_EXAMPLE
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
# YAML discovery and parsing
# ------------------------------------------------------------------


def _find_data_yaml(root: Path) -> Optional[Path]:
    """Return the first ``data*.yaml`` / ``data*.yml`` in *root*."""
    for pattern in ("data*.yaml", "data*.yml"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _resolve_dataset_root(yaml_path: Path) -> Path:
    """Determine the dataset root from a YAML file.

    If the YAML contains a ``path`` entry, it is resolved relative to
    the YAML file's own directory.  Otherwise (or if ``path`` is
    ``"."``), the dataset root is the YAML's parent directory.
    """
    yaml_parent = yaml_path.parent.resolve()

    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        return yaml_parent

    raw_path = data.get("path")
    if not raw_path or str(raw_path).strip() in (".", ""):
        return yaml_parent

    candidate = (yaml_parent / str(raw_path)).resolve()
    if candidate.is_dir():
        return candidate

    logger.warning(
        "The 'path' entry in %s points to '%s' which does not exist; "
        "falling back to the YAML file's parent directory.",
        yaml_path, raw_path,
    )
    return yaml_parent


def _parse_class_names(yaml_path: Path) -> Dict[int, str]:
    """Read class names from a YOLO data YAML."""
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)

    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


_KNOWN_SPLIT_KEYS = ("train", "val", "valid", "test")


def _splits_from_yaml(
    yaml_path: Path, root: Path
) -> Optional[Dict[str, YOLOSplit]]:
    """Build splits by reading image sub-paths from *yaml_path*.

    The YAML file contains keys like ``train: images/train`` or
    ``val: images/val``.  Labels are derived by replacing the
    first occurrence of ``images`` with ``labels`` in the path.
    """
    with open(yaml_path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        return None

    splits: Dict[str, YOLOSplit] = {}

    for key in _KNOWN_SPLIT_KEYS:
        rel_path = data.get(key)
        if not rel_path:
            continue
        rel_path = str(rel_path)

        # Resolve image directory — try relative to root first,
        # then relative to yaml's parent (covers decoupled yaml case).
        img_dir = _resolve_split_dir(rel_path, root, yaml_path.parent)
        if img_dir is None:
            logger.debug(
                "Split '%s' path '%s' not found on disk, skipping", key, rel_path
            )
            continue

        # Derive label directory: replace first "images" with "labels"
        label_rel = rel_path.replace("images", "labels", 1)
        lbl_dir = _resolve_split_dir(label_rel, root, yaml_path.parent)

        split_name = _normalise_split(key)
        image_paths = _find_images(img_dir)
        label_paths = _match_labels(image_paths, lbl_dir) if lbl_dir else []

        splits[split_name] = YOLOSplit(
            name=split_name,
            image_paths=image_paths,
            label_paths=label_paths,
        )

    return splits or None


def _resolve_split_dir(
    rel_path: str, root: Path, yaml_parent: Path
) -> Optional[Path]:
    """Try to resolve *rel_path* relative to root, then yaml parent."""
    candidate = root / rel_path
    if candidate.is_dir():
        return candidate.resolve()
    candidate = yaml_parent / rel_path
    if candidate.is_dir():
        return candidate.resolve()
    return None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _find_images(directory: Path) -> List[Path]:
    """Return sorted list of image files in *directory*."""
    images: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(images))


def _match_labels(
    image_paths: List[Path], label_dir: Optional[Path]
) -> List[Path]:
    """Match each image to its corresponding ``.txt`` label file."""
    labels: List[Path] = []
    if not label_dir or not label_dir.is_dir():
        return labels
    for img in image_paths:
        lbl = label_dir / (img.stem + ".txt")
        if lbl.exists():
            labels.append(lbl)
    return labels
