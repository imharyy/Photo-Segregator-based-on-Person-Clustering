"""
Confidence Heatmap: Generates a 2D UMAP visualization of face embeddings
colored by cluster assignment, with confidence scores shown as size/alpha.
Essential for visual debugging of clustering quality.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("photo_segregator.confidence_heatmap")


class ConfidenceHeatmap:
    """
    Creates a publication-quality 2D scatter plot of face embeddings:
    - UMAP projection to 2D
    - Points colored by cluster assignment
    - Point size proportional to confidence score
    - Review-queue items highlighted in red
    - Noise points marked with 'X' markers

    Saved as a PNG for inclusion in reports and debugging.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Args:
            config: Configuration with 'heatmap' section.
            output_dir: Base output directory.
        """
        hm_cfg = config.get("heatmap", {})
        self.enabled = hm_cfg.get("enabled", True)
        self.output_path = Path(output_dir) / hm_cfg.get(
            "output_path", "_metadata/cluster_heatmap.png"
        )
        self.figsize = tuple(hm_cfg.get("figsize", [14, 10]))
        self.dpi = hm_cfg.get("dpi", 150)

    def generate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        face_ids: List[str],
        review_queue: List[str],
        dim_reducer=None,
    ):
        """
        Generate and save the confidence heatmap.

        Args:
            embeddings: (N, D) face embeddings (original or UMAP-reduced).
            labels: Cluster labels per face.
            confidences: Confidence scores per face.
            face_ids: Face IDs.
            review_queue: Face IDs flagged for review.
            dim_reducer: DimensionalityReducer instance for 2D projection.
        """
        if not self.enabled:
            logger.info("Heatmap generation disabled")
            return

        if len(embeddings) < 2:
            logger.warning("Too few points for heatmap visualization")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping heatmap")
            return

        try:
            # Project to 2D
            if dim_reducer is not None:
                coords_2d = dim_reducer.reduce_for_visualization(embeddings, n_components=2)
            else:
                # Simple fallback: PCA to 2D
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2, random_state=42)
                coords_2d = pca.fit_transform(embeddings)

            # Set up figure
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.set_style("darkgrid")

            # Separate noise from clustered points
            noise_mask = labels == -1
            cluster_mask = ~noise_mask
            review_set = set(review_queue)
            review_mask = np.array([fid in review_set for fid in face_ids])

            # Color palette for clusters
            unique_labels = sorted(set(labels) - {-1})
            n_clusters = len(unique_labels)
            if n_clusters > 0:
                palette = sns.color_palette("husl", n_clusters)
                label_to_color = {l: palette[i] for i, l in enumerate(unique_labels)}
            else:
                label_to_color = {}

            # Plot clustered points (non-review)
            for label in unique_labels:
                mask = (labels == label) & ~review_mask
                if not np.any(mask):
                    continue

                color = label_to_color[label]
                sizes = confidences[mask] * 120 + 20  # Scale sizes
                alphas = np.clip(confidences[mask] * 0.8 + 0.2, 0.3, 1.0)

                for j, (x, y, s, a) in enumerate(
                    zip(coords_2d[mask, 0], coords_2d[mask, 1], sizes, alphas)
                ):
                    ax.scatter(x, y, c=[color], s=s, alpha=a, edgecolors="white",
                             linewidths=0.5, zorder=2)

                # Label the cluster centroid
                centroid_x = np.mean(coords_2d[labels == label, 0])
                centroid_y = np.mean(coords_2d[labels == label, 1])
                ax.annotate(
                    f"P{label:03d}",
                    (centroid_x, centroid_y),
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                    zorder=4,
                )

            # Plot review queue items (red highlight)
            review_cluster_mask = review_mask & cluster_mask
            if np.any(review_cluster_mask):
                ax.scatter(
                    coords_2d[review_cluster_mask, 0],
                    coords_2d[review_cluster_mask, 1],
                    c="red",
                    s=80,
                    alpha=0.8,
                    marker="D",
                    edgecolors="darkred",
                    linewidths=1,
                    label="Review Queue",
                    zorder=3,
                )

            # Plot noise points
            if np.any(noise_mask):
                ax.scatter(
                    coords_2d[noise_mask, 0],
                    coords_2d[noise_mask, 1],
                    c="gray",
                    s=40,
                    alpha=0.5,
                    marker="x",
                    linewidths=1.5,
                    label="Noise / Unclustered",
                    zorder=1,
                )

            # Styling
            ax.set_title(
                f"Face Clustering Confidence Heatmap\n"
                f"{n_clusters} clusters • {int(np.sum(noise_mask))} noise • "
                f"{int(np.sum(review_mask))} review items",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("UMAP Dimension 1", fontsize=11)
            ax.set_ylabel("UMAP Dimension 2", fontsize=11)
            ax.legend(loc="upper right", fontsize=10)

            # Add confidence scale info
            fig.text(
                0.02, 0.02,
                f"Point size ∝ confidence score | "
                f"Avg confidence: {np.mean(confidences[cluster_mask]):.2f}" if np.any(cluster_mask) else "",
                fontsize=9,
                style="italic",
                color="gray",
            )

            plt.tight_layout()

            # Save
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(self.output_path), dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Confidence heatmap saved to {self.output_path}")

        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
