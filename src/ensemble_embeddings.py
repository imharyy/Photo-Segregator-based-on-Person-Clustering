"""
Ensemble Embeddings: Extracts multiple embeddings per face (original, flipped,
perturbed) and averages them for a more robust final embedding.
This is the single most impactful accuracy improvement.
"""

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .face_aligner import create_perturbation

logger = logging.getLogger("photo_segregator.ensemble_embeddings")


class EnsembleEmbedder:
    """
    Creates ensemble embeddings by averaging multiple augmented views of each face.

    Process:
    1. Extract embedding from the aligned crop (original)
    2. Extract embedding from the horizontally flipped crop
    3. Extract embeddings from N slightly perturbed crops (rotation + scale jitter)
    4. L2-normalize each, average all, and re-normalize the final embedding

    This produces embeddings that are far more robust to pose, lighting, and
    minor alignment variations.
    """

    def __init__(self, config: Dict[str, Any], extractor):
        """
        Args:
            config: Configuration dict with 'ensemble' section.
            extractor: An initialized EmbeddingExtractor instance.
        """
        ensemble_cfg = config.get("ensemble", {})
        self.enabled = ensemble_cfg.get("enabled", True)
        self.use_flip = ensemble_cfg.get("use_flip", True)
        self.num_perturbations = ensemble_cfg.get("num_perturbations", 2)
        self.rotation_range = ensemble_cfg.get("rotation_range", 5.0)
        self.scale_jitter = ensemble_cfg.get("scale_jitter", 0.05)
        self.extractor = extractor

    def extract_ensemble(
        self,
        aligned_face: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract an ensemble embedding for a single face.

        Args:
            aligned_face: Primary aligned face image (BGR, 112x112).
            original_image: Full original image (needed for perturbation re-alignment).
            landmarks: Original detected landmarks (needed for perturbation re-alignment).

        Returns:
            Averaged and L2-normalized 512-d embedding, or None on failure.
        """
        if not self.enabled:
            return self.extractor.extract(aligned_face)

        embeddings = []

        # 1. Original aligned face embedding
        emb_original = self.extractor.extract(aligned_face)
        if emb_original is not None:
            embeddings.append(emb_original)
        else:
            # If even the original fails, bail out
            logger.warning("Original embedding extraction failed — cannot create ensemble")
            return None

        # 2. Horizontally flipped embedding
        if self.use_flip:
            flipped = cv2.flip(aligned_face, 1)  # Horizontal flip
            emb_flipped = self.extractor.extract(flipped)
            if emb_flipped is not None:
                embeddings.append(emb_flipped)

        # 3. Perturbation embeddings (if original image and landmarks available)
        if original_image is not None and landmarks is not None:
            rng = np.random.default_rng()
            for i in range(self.num_perturbations):
                # Random rotation within range
                rotation = rng.uniform(-self.rotation_range, self.rotation_range)
                # Random scale within range
                scale = 1.0 + rng.uniform(-self.scale_jitter, self.scale_jitter)

                perturbed = create_perturbation(
                    original_image,
                    landmarks,
                    rotation_deg=rotation,
                    scale_factor=scale,
                )
                if perturbed is not None:
                    emb_perturbed = self.extractor.extract(perturbed)
                    if emb_perturbed is not None:
                        embeddings.append(emb_perturbed)

        # Average all embeddings
        if len(embeddings) == 0:
            return None

        stacked = np.stack(embeddings, axis=0)
        ensemble_emb = np.mean(stacked, axis=0)

        # Re-normalize to unit vector
        norm = np.linalg.norm(ensemble_emb)
        if norm > 0:
            ensemble_emb = ensemble_emb / norm
        else:
            logger.warning("Zero-norm ensemble embedding")
            return None

        logger.debug(f"Ensemble embedding created from {len(embeddings)} views")
        return ensemble_emb.astype(np.float32)

    def extract_ensemble_batch(
        self,
        aligned_faces: List[np.ndarray],
        original_images: Optional[List[np.ndarray]] = None,
        landmarks_list: Optional[List[np.ndarray]] = None,
    ) -> List[Optional[np.ndarray]]:
        """
        Extract ensemble embeddings for a batch of faces.

        Args:
            aligned_faces: List of aligned face images.
            original_images: List of corresponding original images (optional).
            landmarks_list: List of corresponding landmarks (optional).

        Returns:
            List of ensemble embeddings (or None for failed ones).
        """
        results = []

        for i, aligned in enumerate(aligned_faces):
            orig = original_images[i] if original_images and i < len(original_images) else None
            lms = landmarks_list[i] if landmarks_list and i < len(landmarks_list) else None

            emb = self.extract_ensemble(aligned, orig, lms)
            results.append(emb)

        valid = sum(1 for e in results if e is not None)
        logger.info(f"Ensemble extraction: {valid}/{len(aligned_faces)} successful")
        return results
