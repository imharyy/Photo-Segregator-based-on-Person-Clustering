"""
Dimensionality Reduction: UMAP projection from 512-d to a lower-dimensional space.
Reduces noise in high-dimensional embeddings and improves HDBSCAN clustering quality.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("photo_segregator.dimensionality_reduction")


class DimensionalityReducer:
    """
    Applies UMAP dimensionality reduction to face embeddings before clustering.

    Why UMAP before HDBSCAN:
    - 512-d embeddings suffer from the curse of dimensionality
    - UMAP preserves local structure (identity neighborhoods)
    - Tight `min_dist=0.0` produces compact clusters ideal for HDBSCAN
    - Reduces computation time for distance calculations
    - Empirically improves clustering purity
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict with 'umap' section.
        """
        umap_cfg = config.get("umap", {})
        self.enabled = umap_cfg.get("enabled", True)
        self.n_components = umap_cfg.get("n_components", 50)
        self.n_neighbors = umap_cfg.get("n_neighbors", 15)
        self.min_dist = umap_cfg.get("min_dist", 0.0)
        self.metric = umap_cfg.get("metric", "cosine")
        self.reducer = None
        self._fitted = False

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP on the embeddings and transform them to lower dimensionality.

        Args:
            embeddings: (N, 512) array of face embeddings.

        Returns:
            (N, n_components) array of reduced embeddings.
        """
        if not self.enabled:
            logger.info("UMAP disabled — returning original embeddings")
            return embeddings

        if embeddings.shape[0] < 3:
            logger.warning(
                f"Too few samples for UMAP ({embeddings.shape[0]}), returning original"
            )
            return embeddings

        try:
            import umap

            # Adjust n_neighbors for small datasets
            effective_neighbors = min(self.n_neighbors, embeddings.shape[0] - 1)
            effective_components = min(self.n_components, embeddings.shape[0] - 1)

            logger.info(
                f"Fitting UMAP: {embeddings.shape[1]}d → {effective_components}d "
                f"(n_neighbors={effective_neighbors}, min_dist={self.min_dist}, metric={self.metric})"
            )

            self.reducer = umap.UMAP(
                n_components=effective_components,
                n_neighbors=effective_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
                random_state=42,  # Reproducibility
                verbose=False,
            )

            reduced = self.reducer.fit_transform(embeddings)
            self._fitted = True

            logger.info(
                f"UMAP complete: {embeddings.shape} → {reduced.shape}"
            )
            return reduced

        except ImportError:
            logger.warning("umap-learn not installed — skipping dimensionality reduction")
            return embeddings
        except Exception as e:
            logger.error(f"UMAP failed: {e}. Returning original embeddings.")
            return embeddings

    def transform(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings using the already-fitted UMAP model.
        Used for incremental processing.

        Args:
            new_embeddings: (M, 512) array of new embeddings.

        Returns:
            (M, n_components) array of reduced embeddings.
        """
        if not self.enabled or not self._fitted or self.reducer is None:
            return new_embeddings

        try:
            return self.reducer.transform(new_embeddings)
        except Exception as e:
            logger.error(f"UMAP transform failed: {e}")
            return new_embeddings

    def reduce_for_visualization(
        self, embeddings: np.ndarray, n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce embeddings to 2D or 3D for visualization (e.g., heatmap).
        Uses a separate UMAP instance to avoid interfering with the clustering reducer.

        Args:
            embeddings: (N, D) array of embeddings (can be original or already reduced).
            n_components: Target dimensionality (2 for 2D plots, 3 for 3D).

        Returns:
            (N, n_components) array.
        """
        if embeddings.shape[0] < 3:
            # Pad with zeros for tiny datasets
            return np.zeros((embeddings.shape[0], n_components))

        try:
            import umap

            effective_neighbors = min(self.n_neighbors, embeddings.shape[0] - 1)

            viz_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=effective_neighbors,
                min_dist=0.1,  # Slightly spread out for visualization
                metric=self.metric,
                random_state=42,
                verbose=False,
            )
            return viz_reducer.fit_transform(embeddings)

        except Exception as e:
            logger.error(f"UMAP visualization reduction failed: {e}")
            return np.zeros((embeddings.shape[0], n_components))
