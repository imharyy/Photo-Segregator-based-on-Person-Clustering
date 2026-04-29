"""
Adaptive Thresholding: Computes dataset-aware thresholds from embedding statistics.
Instead of fixed thresholds, this module analyzes the actual distance distribution
and sets merge/split/review thresholds dynamically.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger("photo_segregator.adaptive_threshold")


class AdaptiveThreshold:
    """
    Computes clustering thresholds from the statistical properties of the
    embedding distance distribution, rather than using fixed values.

    This is critical because optimal thresholds vary significantly across datasets:
    - Datasets with few people need tighter merge thresholds
    - Datasets with many similar-looking people need looser split thresholds
    - The confidence review cutoff should adapt to the dataset's difficulty
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration with 'adaptive_threshold' and 'refinement' sections.
        """
        at_cfg = config.get("adaptive_threshold", {})
        self.enabled = at_cfg.get("enabled", True)
        self.merge_sigma = at_cfg.get("merge_sigma", 1.5)
        self.split_percentile = at_cfg.get("split_percentile", 75)
        self.review_bottom_pct = at_cfg.get("review_bottom_pct", 15)

        # Manual overrides
        self.fixed_merge = at_cfg.get("fixed_merge_threshold")
        self.fixed_split = at_cfg.get("fixed_split_threshold")
        self.fixed_review = at_cfg.get("fixed_review_threshold")

        # Fallback defaults from refinement config
        ref_cfg = config.get("refinement", {})
        self.default_merge = ref_cfg.get("merge_threshold", 0.3)
        self.default_split = ref_cfg.get("split_threshold", 0.8)
        self.default_review = ref_cfg.get("review_threshold", 0.4)

        # Computed thresholds (populated by compute())
        self.thresholds: Dict[str, float] = {}
        self.statistics: Dict[str, float] = {}

    def compute(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute adaptive thresholds from the embedding distance distribution.

        Args:
            embeddings: (N, D) array of face embeddings.

        Returns:
            Dict with keys: merge_threshold, split_threshold, review_threshold,
            reassign_threshold, plus statistics.
        """
        if not self.enabled or embeddings.shape[0] < 3:
            logger.info("Adaptive thresholding disabled or too few samples — using defaults")
            self.thresholds = {
                "merge_threshold": self.default_merge,
                "split_threshold": self.default_split,
                "review_threshold": self.default_review,
                "reassign_threshold": self.default_merge * 2,
            }
            return self.thresholds

        logger.info(f"Computing adaptive thresholds from {embeddings.shape[0]} embeddings...")

        # Compute pairwise cosine distances
        dist_matrix = cosine_distances(embeddings)

        # Extract upper triangle (avoid diagonal and duplicates)
        upper_indices = np.triu_indices_from(dist_matrix, k=1)
        distances = dist_matrix[upper_indices]

        if len(distances) == 0:
            logger.warning("No pairwise distances to compute — using defaults")
            self.thresholds = {
                "merge_threshold": self.default_merge,
                "split_threshold": self.default_split,
                "review_threshold": self.default_review,
                "reassign_threshold": self.default_merge * 2,
            }
            return self.thresholds

        # Compute statistics
        self.statistics = {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "median": float(np.median(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "p5": float(np.percentile(distances, 5)),
            "p25": float(np.percentile(distances, 25)),
            "p50": float(np.percentile(distances, 50)),
            "p75": float(np.percentile(distances, 75)),
            "p95": float(np.percentile(distances, 95)),
            "num_pairs": len(distances),
        }

        # Compute adaptive thresholds
        mean = self.statistics["mean"]
        std = self.statistics["std"]

        # Merge threshold: conservative (low) — only merge very similar clusters
        # mean - sigma * std, but clamped to reasonable range
        adaptive_merge = max(0.1, min(0.5, mean - self.merge_sigma * std))

        # Split threshold: split clusters with intra-cluster distances above this
        adaptive_split = max(0.4, min(1.2, np.percentile(distances, self.split_percentile)))

        # Review threshold: bottom N% of confidence scores trigger review
        # We use the distance at the review_bottom_pct percentile as a proxy
        adaptive_review = max(0.2, min(0.7, np.percentile(distances, self.review_bottom_pct)))

        # Apply manual overrides if set
        merge_t = self.fixed_merge if self.fixed_merge is not None else adaptive_merge
        split_t = self.fixed_split if self.fixed_split is not None else adaptive_split
        review_t = self.fixed_review if self.fixed_review is not None else adaptive_review

        self.thresholds = {
            "merge_threshold": float(merge_t),
            "split_threshold": float(split_t),
            "review_threshold": float(review_t),
            "reassign_threshold": float(merge_t * 2),
        }

        # Log everything for transparency
        logger.info("=== Adaptive Threshold Report ===")
        logger.info(f"  Distance statistics:")
        for key, val in self.statistics.items():
            logger.info(f"    {key}: {val:.4f}" if isinstance(val, float) else f"    {key}: {val}")
        logger.info(f"  Computed thresholds:")
        for key, val in self.thresholds.items():
            logger.info(f"    {key}: {val:.4f}")
        logger.info("=================================")

        return self.thresholds

    def get_threshold(self, name: str) -> float:
        """Get a specific threshold by name, with fallback to defaults."""
        if name in self.thresholds:
            return self.thresholds[name]
        defaults = {
            "merge_threshold": self.default_merge,
            "split_threshold": self.default_split,
            "review_threshold": self.default_review,
            "reassign_threshold": self.default_merge * 2,
        }
        return defaults.get(name, 0.5)

    def get_report(self) -> Dict[str, Any]:
        """Get full report of statistics and thresholds for logging/saving."""
        return {
            "enabled": self.enabled,
            "statistics": self.statistics,
            "thresholds": self.thresholds,
            "config": {
                "merge_sigma": self.merge_sigma,
                "split_percentile": self.split_percentile,
                "review_bottom_pct": self.review_bottom_pct,
            },
        }
