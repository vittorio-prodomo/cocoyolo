"""Image utilities: fast size extraction without full decode."""

import logging
from pathlib import Path
from typing import Tuple, Union

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def get_image_size(image_path: Union[str, Path]) -> Tuple[int, int]:
    """Return ``(width, height)`` without fully decoding the image.

    Tries *pyvips* first (header-only reads), falls back to *cv2*.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        import pyvips
        img = pyvips.Image.new_from_file(str(path), access="sequential")
        return (img.width, img.height)
    except ImportError:
        pass

    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    return (w, h)
