"""I/O utilities: file linking, image file finding, YAML writing."""

import logging
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
# COCO image finding
# ------------------------------------------------------------------


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
