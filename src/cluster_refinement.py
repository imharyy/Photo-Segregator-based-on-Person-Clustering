"""
Cluster Refinement: Post-clustering merge/split rules and confidence scoring.
Reduces false merges and recovers missed clusters.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger("photo_segregator.cluster_refinement")


class ClusterRefiner:
    """
    Post-clustering refinement pipeline:
    1. Merge check: combine clusters whose centroids are very close
    2. Split check: break apart clusters with high intra-cluster variance
    3. Noise reassignment: assign noise points to nearby clusters if confident
    4. Confidence scoring: compute final per-face confidence
    5. Review queue: flag low-confidence assignments for manual review
    """

    def __init__(self, config: Dict[str, Any], adaptive_thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
            config: Configuration dict with 'refinement' section.
            adaptive_thresholds: Computed thresholds from AdaptiveThreshold (overrides config).
        """
        ref_cfg = config.get("refinement", {})

        # Use adaptive thresholds if available, else config defaults
        if adaptive_thresholds:
            self.merge_threshold = adaptive_thresholds.get("merge_threshold", ref_cfg.get("merge_threshold", 0.3))
            self.split_threshold = adaptive_thresholds.get("split_threshold", ref_cfg.get("split_threshold", 0.8))
            self.reassign_threshold = adaptive_thresholds.get("reassign_threshold", ref_cfg.get("reassign_threshold", 0.6))
            self.review_threshold = adaptive_thresholds.get("review_threshold", ref_cfg.get("review_threshold", 0.4))
        else:
            self.merge_threshold = ref_cfg.get("merge_threshold", 0.3)
            self.split_threshold = ref_cfg.get("split_threshold", 0.8)
            self.reassign_threshold = ref_cfg.get("reassign_threshold", 0.6)
            self.review_threshold = ref_cfg.get("review_threshold", 0.4)

    def refine(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        face_ids: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run the full refinement pipeline.

        Args:
            embeddings: (N, D) original (non-UMAP) embeddings for distance computation.
            labels: Cluster labels from clustering engine.
            probabilities: Membership probabilities from clustering engine.
            face_ids: List of face IDs matching the embeddings rows.

        Returns:
            Tuple of:
            - refined_labels: Updated cluster labels
            - confidences: Per-face confidence scores (0-1)
            - review_queue: List of face_ids flagged for manual review
        """
        labels = labels.copy()
        n_samples = len(labels)

        logger.info(f"Starting cluster refinement on {n_samples} faces")
        initial_clusters = len(set(labels) - {-1})

        # Step 1: Merge similar clusters
        labels = self._merge_clusters(embeddings, labels)
        post_merge_clusters = len(set(labels) - {-1})
        if post_merge_clusters != initial_clusters:
            logger.info(f"  Merge: {initial_clusters} → {post_merge_clusters} clusters")

        # Step 2: Split high-variance clusters
        labels = self._split_clusters(embeddings, labels)
        post_split_clusters = len(set(labels) - {-1})
        if post_split_clusters != post_merge_clusters:
            logger.info(f"  Split: {post_merge_clusters} → {post_split_clusters} clusters")

        # Step 3: Reassign noise points
        noise_before = int(np.sum(labels == -1))
        labels = self._reassign_noise(embeddings, labels)
        noise_after = int(np.sum(labels == -1))
        if noise_before != noise_after:
            logger.info(f"  Noise reassignment: {noise_before} → {noise_after} noise points")

        # Step 4: Compute confidence scores
        confidences = self._compute_confidences(embeddings, labels, probabilities)

        # Step 5: Build review queue
        review_queue = self._build_review_queue(confidences, labels, face_ids)

        final_clusters = len(set(labels) - {-1})
        logger.info(
            f"Refinement complete: {final_clusters} clusters, "
            f"{len(review_queue)} faces in review queue"
        )

        return labels, confidences, review_queue

    def _merge_clusters(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Merge clusters whose centroids are within merge_threshold cosine distance.
        Uses agglomerative approach: iteratively merge closest pair.
        """
        unique_labels = sorted(set(labels) - {-1})
        if len(unique_labels) < 2:
            return labels

        # Compute cluster centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = np.mean(embeddings[mask], axis=0, keepdims=True)

        merged = True
        while merged:
            merged = False
            current_labels = sorted(set(labels) - {-1})
            if len(current_labels) < 2:
                break

            # Find closest pair of centroids
            min_dist = float("inf")
            merge_pair = None

            centroid_list = []
            label_list = []
            for label in current_labels:
                mask = labels == label
                centroid_list.append(np.mean(embeddings[mask], axis=0))
                label_list.append(label)

            centroid_matrix = np.array(centroid_list)
            dist_matrix = cosine_distances(centroid_matrix)

            for i in range(len(label_list)):
                for j in range(i + 1, len(label_list)):
                    if dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        merge_pair = (label_list[i], label_list[j])

            if merge_pair and min_dist < self.merge_threshold:
                # Merge: reassign second cluster to first
                labels[labels == merge_pair[1]] = merge_pair[0]
                merged = True
                logger.debug(
                    f"  Merged cluster {merge_pair[1]} into {merge_pair[0]} "
                    f"(centroid distance={min_dist:.4f})"
                )

        return labels

    def _split_clusters(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Split clusters with intra-cluster max distance exceeding split_threshold.
        Uses k-means (k=2) to split into two sub-clusters.
        """
        from sklearn.cluster import KMeans

        unique_labels = sorted(set(labels) - {-1})
        next_label = max(labels) + 1 if len(labels) > 0 else 0

        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]

            if len(cluster_embeddings) < 4:
                continue  # Too small to meaningfully split

            # Check intra-cluster max distance
            dists = cosine_distances(cluster_embeddings)
            max_dist = np.max(dists)

            if max_dist > self.split_threshold:
                # Attempt split with k-means k=2
                try:
                    km = KMeans(n_clusters=2, random_state=42, n_init=10)
                    sub_labels = km.fit_predict(cluster_embeddings)

                    # Only accept split if both sub-clusters have >= 2 members
                    sizes = [np.sum(sub_labels == 0), np.sum(sub_labels == 1)]
                    if min(sizes) >= 2:
                        indices = np.where(mask)[0]
                        for idx, sub_label in zip(indices, sub_labels):
                            if sub_label == 1:
                                labels[idx] = next_label
                        next_label += 1
                        logger.debug(
                            f"  Split cluster {label} (max_dist={max_dist:.4f}): "
                            f"sizes {sizes[0]}, {sizes[1]}"
                        )
                except Exception as e:
                    logger.debug(f"  Split failed for cluster {label}: {e}")

        return labels

    def _reassign_noise(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Attempt to reassign noise points to their nearest cluster centroid
        if the distance is within reassign_threshold.
        """
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels

        unique_labels = sorted(set(labels) - {-1})
        if len(unique_labels) == 0:
            return labels

        # Compute cluster centroids
        centroids = np.array([
            np.mean(embeddings[labels == l], axis=0) for l in unique_labels
        ])

        # For each noise point, find nearest centroid
        noise_indices = np.where(noise_mask)[0]
        noise_embeddings = embeddings[noise_indices]

        if len(noise_embeddings) == 0:
            return labels

        # Cosine distances between noise points and centroids
        dists = cosine_distances(noise_embeddings, centroids)

        for i, idx in enumerate(noise_indices):
            min_dist = np.min(dists[i])
            nearest_cluster = unique_labels[np.argmin(dists[i])]

            if min_dist < self.reassign_threshold:
                labels[idx] = nearest_cluster
                logger.debug(
                    f"  Reassigned noise face {idx} → cluster {nearest_cluster} "
                    f"(dist={min_dist:.4f})"
                )

        return labels

    def _compute_confidences(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Compute final confidence score for each face's cluster assignment.

        confidence = hdbscan_probability * (1 - distance_to_centroid / max_distance)

        Higher = more confident the face belongs to its assigned cluster.
        """
        confidences = np.zeros(len(labels), dtype=float)
        unique_labels = set(labels) - {-1}

        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            centroid = np.mean(cluster_embeddings, axis=0, keepdims=True)

            # Cosine distance to centroid
            dists = cosine_distances(cluster_embeddings, centroid).flatten()
            max_dist = max(np.max(dists), 1e-6)

            # Distance-based confidence component
            dist_confidence = 1.0 - (dists / (max_dist * 1.5))
            dist_confidence = np.clip(dist_confidence, 0.1, 1.0)

            # Combined confidence: probability * distance_confidence
            cluster_probs = probabilities[mask]
            combined = cluster_probs * dist_confidence
            confidences[mask] = combined

        # Noise points get very low confidence
        confidences[labels == -1] = 0.05

        return confidences

    def _build_review_queue(
        self,
        confidences: np.ndarray,
        labels: np.ndarray,
        face_ids: List[str],
    ) -> List[str]:
        """
        Build a list of face_ids that need manual review.
        Criteria: confidence below review_threshold, or noise label.
        """
        review = []
        for i, (face_id, conf, label) in enumerate(zip(face_ids, confidences, labels)):
            if label == -1 or conf < self.review_threshold:
                review.append(face_id)

        return review
