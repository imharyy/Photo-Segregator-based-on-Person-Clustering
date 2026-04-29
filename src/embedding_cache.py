"""
Embedding Cache: Disk-based storage for face embeddings and metadata.
Supports incremental updates — new faces are appended without reprocessing old ones.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .utils import compute_image_hash, ensure_dir

logger = logging.getLogger("photo_segregator.embedding_cache")


class EmbeddingCache:
    """
    Manages persistent storage of face embeddings and associated metadata.

    Storage format:
    - embeddings.npz: NumPy compressed file with embedding matrix
    - face_metadata.json: Maps face_id → {image_path, bbox, landmarks, quality_score, ...}
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Base output directory (will create _metadata/ subdirectory).
        """
        self.metadata_dir = ensure_dir(Path(output_dir) / "_metadata")
        self.embeddings_path = self.metadata_dir / "embeddings.npz"
        self.face_ids_path = self.metadata_dir / "face_ids.json"
        self.metadata_path = self.metadata_dir / "face_metadata.json"

        # In-memory state
        self.embeddings: Optional[np.ndarray] = None  # (N, 512) array
        self.metadata: Dict[str, Dict] = {}            # face_id → metadata dict
        self.face_ids: List[str] = []                  # Ordered list matching embeddings rows
        self._processed_hashes: Set[str] = set()       # Image hashes already processed
        self._loaded = False

    def load(self):
        """Load existing cache from disk if available."""
        if self._loaded:
            return

        # Load embeddings
        if self.embeddings_path.exists() and self.face_ids_path.exists():
            try:
                data = np.load(self.embeddings_path, allow_pickle=False)
                self.embeddings = data["embeddings"]
                with open(self.face_ids_path, "r", encoding="utf-8") as f:
                    self.face_ids = json.load(f)
                logger.info(f"Loaded {len(self.face_ids)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
                self.embeddings = None
                self.face_ids = []

        # Load metadata
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                # Rebuild processed hashes set
                for face_meta in self.metadata.values():
                    if "image_hash" in face_meta:
                        self._processed_hashes.add(face_meta["image_hash"])
                logger.info(f"Loaded metadata for {len(self.metadata)} faces")
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")
                self.metadata = {}

        self._loaded = True

    def save(self):
        """Save current state to disk."""
        try:
            # Save embeddings (numeric data only — no object arrays)
            if self.embeddings is not None and len(self.face_ids) > 0:
                np.savez_compressed(
                    self.embeddings_path,
                    embeddings=self.embeddings,
                )
                # Save face IDs as JSON (avoids pickle / object array issues)
                with open(self.face_ids_path, "w", encoding="utf-8") as f:
                    json.dump(self.face_ids, f)
                logger.info(f"Saved {len(self.face_ids)} embeddings to {self.embeddings_path}")

            # Save metadata
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, default=str)
            logger.info(f"Saved metadata for {len(self.metadata)} faces to {self.metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            raise

    def add_face(
        self,
        face_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ):
        """
        Add a single face embedding and metadata to the cache.

        Args:
            face_id: Unique face identifier.
            embedding: 512-d embedding vector.
            metadata: Face metadata dict (image_path, bbox, landmarks, quality, etc.)
        """
        embedding = embedding.reshape(1, -1)

        if self.embeddings is None:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])

        self.face_ids.append(face_id)
        self.metadata[face_id] = metadata

        if "image_hash" in metadata:
            self._processed_hashes.add(metadata["image_hash"])

    def is_image_processed(self, image_path: str) -> bool:
        """Check if an image has already been processed (by hash)."""
        try:
            img_hash = compute_image_hash(image_path)
            return img_hash in self._processed_hashes
        except Exception:
            return False

    def get_processed_hashes(self) -> Set[str]:
        """Return set of all processed image hashes."""
        return self._processed_hashes.copy()

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get all cached embeddings and their face IDs.

        Returns:
            Tuple of (embeddings array (N, 512), face_ids list).
        """
        if self.embeddings is None:
            return np.array([]).reshape(0, 512), []
        return self.embeddings, self.face_ids

    def get_embedding(self, face_id: str) -> Optional[np.ndarray]:
        """Get a single embedding by face ID."""
        if face_id in self.face_ids:
            idx = self.face_ids.index(face_id)
            return self.embeddings[idx]
        return None

    def get_metadata(self, face_id: str) -> Optional[Dict]:
        """Get metadata for a specific face."""
        return self.metadata.get(face_id)

    def update_cluster_labels(self, labels: Dict[str, int], confidences: Dict[str, float]):
        """
        Update cluster labels and confidence scores in metadata.

        Args:
            labels: Dict mapping face_id → cluster_label.
            confidences: Dict mapping face_id → confidence_score.
        """
        for face_id, label in labels.items():
            if face_id in self.metadata:
                self.metadata[face_id]["cluster_label"] = int(label)
                self.metadata[face_id]["cluster_confidence"] = float(
                    confidences.get(face_id, 0.0)
                )

    @property
    def count(self) -> int:
        """Number of cached face embeddings."""
        return len(self.face_ids)

    def clear(self):
        """Clear all cached data."""
        self.embeddings = None
        self.metadata = {}
        self.face_ids = []
        self._processed_hashes = set()
        logger.info("Embedding cache cleared")
