"""
Image Loader: Recursively discovers and loads JPG/PNG images from a folder.
Supports incremental mode by skipping already-processed images.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("photo_segregator.image_loader")

# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def discover_images(
    input_dir: str,
    recursive: bool = True,
    skip_hashes: Optional[Set[str]] = None,
) -> List[Path]:
    """
    Discover all supported image files in the input directory.

    Args:
        input_dir: Path to the input folder.
        recursive: Whether to scan subdirectories.
        skip_hashes: Set of image hashes to skip (for incremental mode).

    Returns:
        List of Path objects for discovered images.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    image_paths = []
    pattern_func = input_path.rglob if recursive else input_path.glob

    for ext in SUPPORTED_EXTENSIONS:
        # Case-insensitive matching by checking both cases
        for path in pattern_func(f"*{ext}"):
            if path.is_file():
                image_paths.append(path)
        for path in pattern_func(f"*{ext.upper()}"):
            if path.is_file() and path not in image_paths:
                image_paths.append(path)

    # Sort for deterministic processing order
    image_paths.sort()

    logger.info(f"Discovered {len(image_paths)} images in '{input_dir}' (recursive={recursive})")
    return image_paths


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """
    Load a single image using OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        BGR image array, or None if the image is corrupt/unreadable.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image (corrupt or unsupported): {image_path}")
            return None
        if img.size == 0:
            logger.warning(f"Empty image: {image_path}")
            return None
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def load_images_batch(
    image_paths: List[Path],
) -> List[Tuple[Path, np.ndarray]]:
    """
    Load a batch of images, skipping any that fail to load.

    Args:
        image_paths: List of image paths to load.

    Returns:
        List of (path, image_array) tuples for successfully loaded images.
    """
    loaded = []
    failed_count = 0

    for path in image_paths:
        img = load_image(path)
        if img is not None:
            loaded.append((path, img))
        else:
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"Failed to load {failed_count}/{len(image_paths)} images")

    logger.info(f"Successfully loaded {len(loaded)} images")
    return loaded


def get_image_info(image_path: Path) -> Dict:
    """
    Get basic information about an image without fully loading it.

    Returns:
        Dict with keys: path, filename, extension, size_bytes
    """
    stat = image_path.stat()
    return {
        "path": str(image_path.resolve()),
        "filename": image_path.name,
        "extension": image_path.suffix.lower(),
        "size_bytes": stat.st_size,
    }
