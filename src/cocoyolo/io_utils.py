"""I/O utilities: file linking, image file finding, YAML writing, RLE decoding."""

import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# File linking with fallback
# ------------------------------------------------------------------


VALID_IMAGE_MODES = ("copy", "symlink", "hardlink")


class FileLinker:
    """Transfer files from source to destination using the chosen mode.

    Supported modes:

    - ``"copy"`` (default) — full file copy via ``shutil.copy2``.
    - ``"symlink"`` — relative symbolic link (falls back to copy if the OS
      does not support symlinks, e.g. Windows without developer mode).
    - ``"hardlink"`` — hard link (falls back to copy if the OS does not
      support hard links or if source and destination are on different
      filesystems).

    Tracks all operations for atomic rollback on errors.
    """

    def __init__(self, mode: str = "copy") -> None:
        if mode not in VALID_IMAGE_MODES:
            raise ValueError(
                f"image_mode must be one of {VALID_IMAGE_MODES}, got {mode!r}"
            )
        self._mode = mode
        self._log: List[Tuple[str, Path]] = []

        if mode == "symlink":
            self._symlinks_ok = _check_symlink_support()
            if not self._symlinks_ok:
                logger.warning(
                    "Symlinks not supported on this platform; "
                    "falling back to copy."
                )
        elif mode == "hardlink":
            self._hardlinks_ok = _check_hardlink_support()
            if not self._hardlinks_ok:
                logger.warning(
                    "Hard links not supported on this platform; "
                    "falling back to copy."
                )

    def link(self, src: Path, dst: Path) -> None:
        """Transfer *src* to *dst* using the configured mode."""
        if not src.exists():
            raise FileNotFoundError(f"Source file does not exist: {src}")
        if src.resolve() == dst.resolve():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)

        if self._mode == "symlink" and self._symlinks_ok:
            try:
                rel = os.path.relpath(src, dst.parent)
                os.symlink(rel, dst)
                self._log.append(("symlink", dst))
                return
            except (OSError, ValueError):
                logger.debug("Symlink failed for %s; falling back to copy.", dst)

        elif self._mode == "hardlink" and self._hardlinks_ok:
            try:
                os.link(src.resolve(), dst)
                self._log.append(("hardlink", dst))
                return
            except OSError:
                logger.debug("Hardlink failed for %s; falling back to copy.", dst)

        shutil.copy2(src, dst)
        self._log.append(("copy", dst))

    def rollback(self) -> None:
        """Undo every logged operation in reverse order."""
        for _, dst in reversed(self._log):
            try:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
            except Exception as exc:
                logger.warning("Rollback failed for %s: %s", dst, exc)
        self._log.clear()


def _check_symlink_support() -> bool:
    """Return True if the OS supports creating symbolic links."""
    if platform.system() != "Windows":
        return True
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            lnk = Path(tmp) / "lnk.txt"
            src.write_text("test")
            os.symlink(src, lnk)
            lnk.unlink()
        return True
    except (OSError, NotImplementedError):
        return False


def _check_hardlink_support() -> bool:
    """Return True if the OS supports creating hard links."""
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            lnk = Path(tmp) / "lnk.txt"
            src.write_text("test")
            os.link(src, lnk)
            lnk.unlink()
        return True
    except (OSError, NotImplementedError):
        return False


# ------------------------------------------------------------------
# Image index / finding
# ------------------------------------------------------------------


def build_image_index(
    search_root: Union[str, Path],
    image_extensions: Optional[Set[str]] = None,
) -> Dict[str, Path]:
    """Build a ``{filename: path}`` index by scanning *search_root* once.

    If two files share the same name in different sub-directories, a
    ``ValueError`` is raised listing all duplicates.  Duplicate filenames
    are a recipe for silent data corruption in downstream pipelines
    (both COCO and YOLO resolve images by basename only).

    Args:
        search_root: Root directory to scan recursively.
        image_extensions: If given, only index files with these suffixes
            (case-insensitive).  Pass ``None`` to index all files.

    Returns:
        Mapping from filename to its resolved path.
    """
    root = Path(search_root)
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if image_extensions is not None and p.suffix.lower() not in image_extensions:
            continue
        name = p.name
        if name in index:
            if name not in duplicates:
                duplicates[name] = [index[name]]
            duplicates[name].append(p)
        else:
            index[name] = p

    if duplicates:
        lines = ["Duplicate filenames found under " + str(root) + ":"]
        for name, paths in sorted(duplicates.items()):
            lines.append(f"  {name}:")
            for dp in paths:
                lines.append(f"    - {dp}")
        lines.append(
            "\nDuplicate filenames within a dataset cause ambiguous image "
            "matching and must be resolved before conversion."
        )
        raise ValueError("\n".join(lines))

    return index


def find_coco_image(
    filename: str, search_root: Union[str, Path]
) -> Optional[Path]:
    """Recursively find *filename* under *search_root*.

    Returns the first match, or ``None``.
    """
    root = Path(search_root)
    for p in root.rglob(filename):
        if p.is_file():
            return p
    return None


# ------------------------------------------------------------------
# YAML writing
# ------------------------------------------------------------------


def create_data_yaml(
    output_path: Path,
    classes: Union[List[str], Dict[int, str]],
    splits: Union[List[str], Set[str]],
) -> Path:
    """Create a YOLO ``data.yaml`` file.

    Args:
        output_path: Directory where ``data.yaml`` will be written.
        classes: Class names as a list (0-indexed) or ``{id: name}`` dict.
        splits: Split names.

    Returns:
        Path to the created ``data.yaml`` file.
    """
    output_path = Path(output_path)

    if isinstance(classes, list):
        class_dict = {i: name for i, name in enumerate(classes)}
    elif isinstance(classes, dict):
        class_dict = {int(k): str(v) for k, v in classes.items()}
    else:
        raise TypeError(f"classes must be a list or dict, got {type(classes)}")

    split_names = sorted(splits) if isinstance(splits, set) else list(splits)

    yaml_data: Dict[str, Any] = {
        "nc": len(class_dict),
        "names": class_dict,
    }
    for name in split_names:
        yaml_data[name] = f"images/{name}"

    yaml_path = output_path / "data.yaml"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)

    logger.info("Created data.yaml with %d classes at %s", len(class_dict), yaml_path)
    return yaml_path


# ------------------------------------------------------------------
# RLE decoding
# ------------------------------------------------------------------


def decode_rle(rle_data: Dict) -> np.ndarray:
    """Decode a COCO RLE (compressed or uncompressed) to a binary mask.

    Args:
        rle_data: A dict with ``"counts"`` and ``"size"`` keys, as produced
            by the COCO annotation format.  Both uncompressed (list of ints)
            and compressed (string) count formats are supported.

    Returns:
        A 2-D ``uint8`` NumPy array (H x W) with 1 for foreground pixels.

    Raises:
        ValueError: If *rle_data* is not a dict or is missing required keys.
    """
    from pycocotools import mask as coco_mask_util

    if not isinstance(rle_data, dict):
        raise ValueError(f"RLE data must be a dict, got {type(rle_data)}")
    if "counts" not in rle_data or "size" not in rle_data:
        raise ValueError("RLE data must contain 'counts' and 'size' keys")

    if isinstance(rle_data["counts"], list):
        rle = coco_mask_util.frPyObjects(
            rle_data, rle_data["size"][0], rle_data["size"][1]
        )
        return coco_mask_util.decode(rle)

    return coco_mask_util.decode(rle_data)
