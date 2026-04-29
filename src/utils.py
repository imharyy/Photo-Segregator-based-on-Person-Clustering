"""
Utility functions: config loading, logging setup, image hashing, helpers.
"""

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with defaults."""
    config_file = Path(config_path)
    if not config_file.exists():
        logging.warning(f"Config file not found: {config_path}. Using defaults.")
        return _default_config()

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Merge with defaults so missing keys don't crash
    defaults = _default_config()
    return _deep_merge(defaults, config)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        "input_dir": "./photos",
        "output_dir": "./output",
        "copy_mode": "copy",
        "save_crops": True,
        "recursive": True,
        "detection": {
            "model_pack": "buffalo_l",
            "det_size": [640, 640],
            "det_thresh": 0.5,
            "max_faces": 50,
        },
        "quality": {
            "min_face_size": 40,
            "blur_threshold": 50.0,
            "max_pose_angle": 75.0,
            "min_landmark_conf": 0.3,
        },
        "embedding": {
            "dimension": 512,
            "normalize": True,
        },
        "ensemble": {
            "enabled": True,
            "use_flip": True,
            "num_perturbations": 2,
            "rotation_range": 5.0,
            "scale_jitter": 0.05,
        },
        "umap": {
            "enabled": True,
            "n_components": 50,
            "n_neighbors": 15,
            "min_dist": 0.0,
            "metric": "cosine",
        },
        "clustering": {
            "primary_method": "hdbscan",
            "fallback_method": "dbscan",
            "noise_fallback_ratio": 0.6,
            "hdbscan": {
                "min_cluster_size": 2,
                "min_samples": 1,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "prediction_data": True,
            },
            "dbscan": {
                "eps": 0.5,
                "min_samples": 2,
                "metric": "euclidean",
            },
        },
        "adaptive_threshold": {
            "enabled": True,
            "merge_sigma": 1.5,
            "split_percentile": 75,
            "review_bottom_pct": 15,
            "fixed_merge_threshold": None,
            "fixed_split_threshold": None,
            "fixed_review_threshold": None,
        },
        "refinement": {
            "merge_threshold": 0.3,
            "split_threshold": 0.8,
            "reassign_threshold": 0.6,
            "review_threshold": 0.4,
        },
        "semi_supervised": {
            "enabled": True,
            "known_identities_path": "_metadata/known_identities.npz",
            "classification_threshold": 0.4,
        },
        "heatmap": {
            "enabled": True,
            "output_path": "_metadata/cluster_heatmap.png",
            "figsize": [14, 10],
            "dpi": 150,
        },
        "logging": {
            "level": "INFO",
            "log_dir": "_logs",
            "log_file": "pipeline.log",
            "console": True,
        },
    }


def setup_logging(config: Dict[str, Any], output_dir: str) -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    Returns the root logger for the pipeline.
    """
    log_cfg = config.get("logging", {})
    log_dir = Path(output_dir) / log_cfg.get("log_dir", "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / log_cfg.get("log_file", "pipeline.log")
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("photo_segregator")
    logger.setLevel(level)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    if log_cfg.get("console", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def compute_image_hash(image_path: str, algorithm: str = "sha256") -> str:
    """
    Compute a hash of an image file for deduplication and cache keys.
    Uses file content hash (not filename) for robustness.
    """
    hash_func = hashlib.new(algorithm)
    with open(image_path, "rb") as f:
        # Read in chunks for memory efficiency with large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_face_id(image_hash: str, face_index: int) -> str:
    """Generate a unique face ID from image hash and face index."""
    return f"{image_hash[:16]}_{face_index:03d}"
