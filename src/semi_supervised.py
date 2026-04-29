"""
Semi-Supervised Learning: Uses corrected labels from manual review to improve
future clustering by learning known identity centroids.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger("photo_segregator.semi_supervised")


class SemiSupervisedLearner:
    """
    After manual review, uses corrected cluster labels to:
    1. Compute per-identity centroid embeddings
    2. Store them as "known identities"
    3. In future runs, classify new faces against known identities before clustering
    4. Only cluster truly unknown faces

    This creates a feedback loop: review → learn → better clustering next time.
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Args:
            config: Configuration with 'semi_supervised' section.
            output_dir: Base output directory.
        """
        ss_cfg = config.get("semi_supervised", {})
        self.enabled = ss_cfg.get("enabled", True)
        self.threshold = ss_cfg.get("classification_threshold", 0.4)
        self.identities_path = Path(output_dir) / ss_cfg.get(
            "known_identities_path", "_metadata/known_identities.npz"
        )

        # Known identities: {identity_id: centroid_embedding}
        self.known_centroids: Optional[np.ndarray] = None
        self.known_ids: List[str] = []
        self._loaded = False

    def load(self):
        """Load known identities from disk if available."""
        if self._loaded:
            return

        if self.identities_path.exists():
            try:
                data = np.load(self.identities_path, allow_pickle=True)
                self.known_centroids = data["centroids"]
                self.known_ids = data["identity_ids"].tolist()
                logger.info(f"Loaded {len(self.known_ids)} known identities")
            except Exception as e:
                logger.warning(f"Failed to load known identities: {e}")
                self.known_centroids = None
                self.known_ids = []

        self._loaded = True

    def save(self):
        """Save known identities to disk."""
        if self.known_centroids is None or len(self.known_ids) == 0:
            return

        try:
            self.identities_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.identities_path,
                centroids=self.known_centroids,
                identity_ids=np.array(self.known_ids, dtype=object),
            )
            logger.info(f"Saved {len(self.known_ids)} known identities to {self.identities_path}")
        except Exception as e:
            logger.error(f"Failed to save known identities: {e}")

    def learn_from_corrections(
        self,
        embeddings: np.ndarray,
        face_ids: List[str],
        corrected_labels: Dict[str, int],
    ):
        """
        Learn identity centroids from corrected labels.

        Args:
            embeddings: (N, D) face embeddings.
            face_ids: Face IDs matching embeddings rows.
            corrected_labels: Dict mapping face_id → corrected cluster label.
        """
        if not self.enabled:
            return

        # Build label array
        face_to_idx = {fid: i for i, fid in enumerate(face_ids)}
        labels = {}
        for face_id, label in corrected_labels.items():
            if face_id in face_to_idx and label >= 0:
                idx = face_to_idx[face_id]
                if label not in labels:
                    labels[label] = []
                labels[label].append(idx)

        if not labels:
            logger.info("No valid corrections to learn from")
            return

        # Compute centroids for each identity
        new_centroids = []
        new_ids = []

        for label, indices in sorted(labels.items()):
            cluster_embeddings = embeddings[indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            # Normalize centroid
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            identity_id = f"Person_{label:03d}"
            new_centroids.append(centroid)
            new_ids.append(identity_id)

        # Merge with existing known identities
        if self.known_centroids is not None and len(self.known_ids) > 0:
            # Check if any new identities match existing ones
            new_centroid_matrix = np.array(new_centroids)
            dists = cosine_distances(new_centroid_matrix, self.known_centroids)

            merged_centroids = list(self.known_centroids)
            merged_ids = list(self.known_ids)

            for i, (new_c, new_id) in enumerate(zip(new_centroids, new_ids)):
                min_dist = np.min(dists[i])
                closest_idx = np.argmin(dists[i])

                if min_dist < self.threshold:
                    # Update existing identity centroid (running average)
                    old = merged_centroids[closest_idx]
                    merged_centroids[closest_idx] = (old + new_c) / 2.0
                    # Re-normalize
                    norm = np.linalg.norm(merged_centroids[closest_idx])
                    if norm > 0:
                        merged_centroids[closest_idx] = merged_centroids[closest_idx] / norm
                    logger.debug(f"Updated known identity: {merged_ids[closest_idx]}")
                else:
                    merged_centroids.append(new_c)
                    merged_ids.append(new_id)
                    logger.debug(f"Added new identity: {new_id}")

            self.known_centroids = np.array(merged_centroids)
            self.known_ids = merged_ids
        else:
            self.known_centroids = np.array(new_centroids)
            self.known_ids = new_ids

        logger.info(f"Learned from corrections: {len(labels)} identities → {len(self.known_ids)} total known")
        self.save()

    def classify_known(
        self, embeddings: np.ndarray, face_ids: List[str]
    ) -> Tuple[Dict[str, int], List[int]]:
        """
        Classify embeddings against known identities.
        Returns labels for confidently matched faces, and indices of unknowns.

        Args:
            embeddings: (N, D) face embeddings.
            face_ids: Face IDs matching embeddings rows.

        Returns:
            Tuple of:
            - known_labels: Dict mapping face_id → cluster_label for classified faces
            - unknown_indices: List of indices for faces that couldn't be classified
        """
        self.load()

        if not self.enabled or self.known_centroids is None or len(self.known_ids) == 0:
            # No known identities: everything is unknown
            return {}, list(range(len(face_ids)))

        # Compute distances between all embeddings and known centroids
        dists = cosine_distances(embeddings, self.known_centroids)

        known_labels = {}
        unknown_indices = []

        for i, face_id in enumerate(face_ids):
            min_dist = np.min(dists[i])
            nearest_identity = np.argmin(dists[i])

            if min_dist < self.threshold:
                known_labels[face_id] = nearest_identity
                logger.debug(
                    f"Classified {face_id} as {self.known_ids[nearest_identity]} "
                    f"(dist={min_dist:.4f})"
                )
            else:
                unknown_indices.append(i)

        classified_count = len(known_labels)
        logger.info(
            f"Semi-supervised classification: {classified_count} known, "
            f"{len(unknown_indices)} unknown out of {len(face_ids)}"
        )

        return known_labels, unknown_indices

    @property
    def has_known_identities(self) -> bool:
        """Check if any known identities are loaded."""
        self.load()
        return self.known_centroids is not None and len(self.known_ids) > 0
