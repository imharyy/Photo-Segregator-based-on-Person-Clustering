"""
Embedding Extractor: Extracts 512-d ArcFace embeddings from aligned face images.
Uses InsightFace's built-in recognition model from the buffalo_l pack.
"""

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("photo_segregator.embedding_extractor")


class EmbeddingExtractor:
    """
    Extracts normalized 512-dimensional face embeddings using ArcFace.
    Leverages the InsightFace FaceAnalysis recognition model.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the embedding extractor.

        Args:
            config: Configuration dict with 'embedding' section.
        """
        self.config = config.get("embedding", {})
        self.dimension = self.config.get("dimension", 512)
        self.normalize = self.config.get("normalize", True)
        self.rec_model = None
        self._initialized = False

        # Store the detection config for model initialization
        self._detection_config = config.get("detection", {})

    def initialize(self):
        """Initialize the recognition model. Lazy — called on first use."""
        if self._initialized:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis

            model_pack = self._detection_config.get("model_pack", "buffalo_l")

            # Try GPU first, fallback to CPU
            providers = self._get_providers()

            # Initialize FaceAnalysis to get the recognition model
            app = FaceAnalysis(name=model_pack, providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))

            # Extract the recognition model from the app
            self.rec_model = None
            for model in app.models.values():
                if hasattr(model, 'get_feat') or (hasattr(model, 'taskname') and model.taskname == 'recognition'):
                    self.rec_model = model
                    break

            # Fallback: try to find by input shape
            if self.rec_model is None:
                for model in app.models.values():
                    if hasattr(model, 'input_size') and model.input_size == (112, 112):
                        self.rec_model = model
                        break

            if self.rec_model is None:
                raise RuntimeError("Could not find recognition model in the InsightFace model pack")

            self._initialized = True
            logger.info(f"EmbeddingExtractor initialized (dim={self.dimension})")

        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingExtractor: {e}")
            raise RuntimeError(f"EmbeddingExtractor initialization failed: {e}") from e

    def _get_providers(self) -> List[str]:
        """Detect available ONNX Runtime execution providers."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            return providers
        except ImportError:
            return ["CPUExecutionProvider"]

    def extract(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-d embedding from a single aligned face image.

        Args:
            aligned_face: Aligned face image (BGR, 112x112).

        Returns:
            512-d numpy array (L2-normalized if configured), or None on failure.
        """
        self.initialize()

        try:
            # Ensure correct size
            if aligned_face.shape[:2] != (112, 112):
                aligned_face = cv2.resize(aligned_face, (112, 112))

            # The recognition model expects a specific format
            # InsightFace models typically work with the face object,
            # but we can call get_feat directly on prepared input
            embedding = self._compute_embedding(aligned_face)

            if embedding is None:
                return None

            # Ensure correct dimensionality
            embedding = embedding.flatten()
            if len(embedding) != self.dimension:
                logger.warning(
                    f"Unexpected embedding dimension: {len(embedding)} (expected {self.dimension})"
                )

            # L2 normalize
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                else:
                    logger.warning("Zero-norm embedding detected")
                    return None

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def _compute_embedding(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute raw embedding using the recognition model.

        InsightFace's ArcFaceONNX.get_feat() expects a list of (H, W, C) uint8 images.
        It handles preprocessing (mean subtraction, scaling, CHW transpose) internally.
        """
        try:
            # Method 1: get_feat with list of images (standard InsightFace API)
            if hasattr(self.rec_model, 'get_feat'):
                # get_feat expects a list of BGR uint8 images, NOT a pre-processed blob
                embedding = self.rec_model.get_feat([aligned_face])
                return embedding

            # Method 2: Direct ONNX session inference (fallback)
            if hasattr(self.rec_model, 'session'):
                input_mean = getattr(self.rec_model, 'input_mean', 127.5)
                input_std = getattr(self.rec_model, 'input_std', 127.5)

                # Manually preprocess: BGR→RGB, HWC→CHW, normalize
                img = aligned_face.astype(np.float32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = (img - input_mean) / input_std
                img = np.transpose(img, (2, 0, 1))  # HWC → CHW
                blob = np.expand_dims(img, axis=0).astype(np.float32)

                input_name = self.rec_model.session.get_inputs()[0].name
                output = self.rec_model.session.run(None, {input_name: blob})
                return output[0]

            logger.error("Recognition model has no recognized inference method")
            return None

        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return None

    def extract_batch(self, aligned_faces: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from a batch of aligned faces.

        Args:
            aligned_faces: List of aligned face images (BGR, 112x112).

        Returns:
            List of 512-d embeddings (or None for failed extractions).
        """
        self.initialize()
        embeddings = []

        for face in aligned_faces:
            emb = self.extract(face)
            embeddings.append(emb)

        valid_count = sum(1 for e in embeddings if e is not None)
        logger.info(f"Extracted {valid_count}/{len(aligned_faces)} embeddings")
        return embeddings
