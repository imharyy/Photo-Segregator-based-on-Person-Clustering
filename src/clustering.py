"""
Clustering Engine: HDBSCAN (primary) with DBSCAN fallback.
Performs density-based clustering on face embeddings.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("photo_segregator.clustering")


class ClusteringEngine:
    """
    Clusters face embeddings using HDBSCAN as the primary method
    and DBSCAN as a fallback when HDBSCAN produces too much noise.

    Why HDBSCAN:
    - No need to specify number of clusters
    - Handles noise natively (labels outliers as -1)
    - Provides per-point membership probabilities for confidence scoring
    - Excess of Mass (EOM) cluster selection is stable

    Why DBSCAN fallback:
    - When HDBSCAN over-fragments (>60% noise), DBSCAN with a fixed eps
      can sometimes recover better structure
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict with 'clustering' section.
        """
        cluster_cfg = config.get("clustering", {})
        self.primary = cluster_cfg.get("primary_method", "hdbscan")
        self.fallback = cluster_cfg.get("fallback_method", "dbscan")
        self.noise_fallback_ratio = cluster_cfg.get("noise_fallback_ratio", 0.6)

        # HDBSCAN params
        hdb_cfg = cluster_cfg.get("hdbscan", {})
        self.hdb_min_cluster_size = hdb_cfg.get("min_cluster_size", 2)
        self.hdb_min_samples = hdb_cfg.get("min_samples", 1)
        self.hdb_metric = hdb_cfg.get("metric", "euclidean")
        self.hdb_selection = hdb_cfg.get("cluster_selection_method", "eom")
        self.hdb_prediction_data = hdb_cfg.get("prediction_data", True)

        # DBSCAN params
        db_cfg = cluster_cfg.get("dbscan", {})
        self.db_eps = db_cfg.get("eps", 0.5)
        self.db_min_samples = db_cfg.get("min_samples", 2)
        self.db_metric = db_cfg.get("metric", "euclidean")

        # Results
        self.labels: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.method_used: str = ""

    def cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings using HDBSCAN, with DBSCAN fallback.

        Args:
            embeddings: (N, D) array of (possibly UMAP-reduced) embeddings.

        Returns:
            Tuple of (labels, probabilities).
            Labels: integer cluster IDs (-1 = noise).
            Probabilities: membership confidence (0-1) per point.
        """
        n_samples = embeddings.shape[0]

        if n_samples == 0:
            logger.warning("No embeddings to cluster")
            return np.array([], dtype=int), np.array([], dtype=float)

        if n_samples == 1:
            logger.info("Single embedding — assigning to cluster 0")
            return np.array([0]), np.array([1.0])

        # Try primary method (HDBSCAN)
        logger.info(f"Clustering {n_samples} embeddings with {self.primary.upper()}")
        labels, probs = self._run_hdbscan(embeddings)

        # Check noise ratio
        noise_count = np.sum(labels == -1)
        noise_ratio = noise_count / len(labels) if len(labels) > 0 else 0
        n_clusters = len(set(labels) - {-1})

        logger.info(
            f"HDBSCAN result: {n_clusters} clusters, "
            f"{noise_count}/{len(labels)} noise points ({noise_ratio:.1%})"
        )

        # Fallback to DBSCAN if too much noise
        if noise_ratio > self.noise_fallback_ratio and self.fallback == "dbscan":
            logger.warning(
                f"HDBSCAN noise ratio {noise_ratio:.1%} exceeds threshold "
                f"{self.noise_fallback_ratio:.1%} — falling back to DBSCAN"
            )
            labels, probs = self._run_dbscan(embeddings)
            noise_count = np.sum(labels == -1)
            n_clusters = len(set(labels) - {-1})
            logger.info(
                f"DBSCAN result: {n_clusters} clusters, "
                f"{noise_count}/{len(labels)} noise points"
            )

        self.labels = labels
        self.probabilities = probs
        return labels, probs

    def _run_hdbscan(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run HDBSCAN clustering using scikit-learn's implementation."""
        try:
            from sklearn.cluster import HDBSCAN

            # Adjust min_cluster_size for very small datasets
            effective_min_cluster = min(
                self.hdb_min_cluster_size, max(2, embeddings.shape[0] // 3)
            )

            clusterer = HDBSCAN(
                min_cluster_size=effective_min_cluster,
                min_samples=self.hdb_min_samples,
                metric=self.hdb_metric,
                cluster_selection_method=self.hdb_selection,
                store_centers="centroid",
            )
            clusterer.fit(embeddings)
            self.method_used = "hdbscan"
            self._clusterer = clusterer  # Save for potential later use

            labels = clusterer.labels_

            # sklearn's HDBSCAN provides probabilities_ when available
            if hasattr(clusterer, 'probabilities_') and clusterer.probabilities_ is not None:
                probs = clusterer.probabilities_
            else:
                # Estimate probabilities from distance to cluster centroids
                probs = self._estimate_dbscan_probabilities(embeddings, labels)

            return labels, probs

        except Exception as e:
            logger.error(f"HDBSCAN failed: {e}")
            # Emergency fallback: each point is its own cluster
            return np.arange(embeddings.shape[0]), np.ones(embeddings.shape[0])

    def _run_dbscan(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run DBSCAN clustering (fallback)."""
        try:
            from sklearn.cluster import DBSCAN

            clusterer = DBSCAN(
                eps=self.db_eps,
                min_samples=self.db_min_samples,
                metric=self.db_metric,
            )
            labels = clusterer.fit_predict(embeddings)
            self.method_used = "dbscan"

            # DBSCAN doesn't provide probabilities, so we estimate from distances
            probs = self._estimate_dbscan_probabilities(embeddings, labels)

            return labels, probs

        except Exception as e:
            logger.error(f"DBSCAN failed: {e}")
            return np.arange(embeddings.shape[0]), np.ones(embeddings.shape[0])

    def _estimate_dbscan_probabilities(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Estimate membership probabilities for DBSCAN results.
        Based on distance to cluster centroid (closer = higher probability).
        """
        probs = np.zeros(len(labels), dtype=float)
        unique_labels = set(labels) - {-1}

        for label in unique_labels:
            mask = labels == label
            cluster_points = embeddings[mask]
            centroid = np.mean(cluster_points, axis=0)

            # Distance to centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            max_dist = max(max_dist, 1e-6)

            # Probability: inversely proportional to distance
            cluster_probs = 1.0 - (distances / (max_dist * 1.5))
            cluster_probs = np.clip(cluster_probs, 0.1, 1.0)
            probs[mask] = cluster_probs

        # Noise points get low probability
        probs[labels == -1] = 0.05

        return probs

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get a summary of the clustering results."""
        if self.labels is None:
            return {"status": "not_clustered"}

        unique_labels = set(self.labels)
        n_clusters = len(unique_labels - {-1})
        noise_count = int(np.sum(self.labels == -1))

        cluster_sizes = {}
        for label in sorted(unique_labels - {-1}):
            cluster_sizes[int(label)] = int(np.sum(self.labels == label))

        return {
            "method": self.method_used,
            "n_clusters": n_clusters,
            "n_noise": noise_count,
            "n_total": len(self.labels),
            "noise_ratio": noise_count / len(self.labels) if len(self.labels) > 0 else 0,
            "cluster_sizes": cluster_sizes,
            "avg_probability": float(np.mean(self.probabilities)) if self.probabilities is not None else 0,
        }
