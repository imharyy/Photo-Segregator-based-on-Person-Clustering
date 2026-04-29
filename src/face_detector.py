"""
Face Detector: Uses RetinaFace (via InsightFace) for face detection and landmark extraction.
Supports GPU (CUDA) with automatic CPU fallback.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("photo_segregator.face_detector")


@dataclass
class DetectedFace:
    """Represents a single detected face in an image."""
    bbox: np.ndarray            # [x1, y1, x2, y2] bounding box
    landmarks: np.ndarray       # 5x2 array: left_eye, right_eye, nose, left_mouth, right_mouth
    det_score: float            # Detection confidence (0-1)
    face_index: int             # Index of this face within the image
    image_path: str             # Path to the source image
    face_crop: Optional[np.ndarray] = field(default=None, repr=False)  # Raw face crop (BGR)

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def bbox_area(self) -> float:
        return self.bbox_width * self.bbox_height

    @property
    def min_side(self) -> float:
        return min(self.bbox_width, self.bbox_height)


class FaceDetector:
    """
    Face detector using RetinaFace from the InsightFace library.
    Handles GPU/CPU selection, model initialization, and batch detection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the face detector.

        Args:
            config: Configuration dictionary with 'detection' section.
        """
        self.config = config.get("detection", {})
        self.model_pack = self.config.get("model_pack", "buffalo_l")
        self.det_size = tuple(self.config.get("det_size", [640, 640]))
        self.det_thresh = self.config.get("det_thresh", 0.5)
        self.max_faces = self.config.get("max_faces", 50)
        self.app = None
        self._initialized = False

    def initialize(self):
        """
        Initialize the InsightFace FaceAnalysis app with GPU/CPU auto-detection.
        Lazy initialization — called on first use.
        """
        if self._initialized:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis

            # Try GPU first, fallback to CPU
            providers = self._get_providers()
            logger.info(f"Initializing FaceAnalysis with model_pack='{self.model_pack}', providers={providers}")

            self.app = FaceAnalysis(
                name=self.model_pack,
                providers=providers,
            )
            self.app.prepare(
                ctx_id=0,
                det_size=self.det_size,
            )
            self._initialized = True
            logger.info(f"FaceDetector initialized successfully (det_size={self.det_size}, det_thresh={self.det_thresh})")

        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {e}")
            raise RuntimeError(f"FaceDetector initialization failed: {e}") from e

    def _get_providers(self) -> List[str]:
        """Detect available ONNX Runtime execution providers."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            logger.info(f"Available ONNX Runtime providers: {available}")

            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
                logger.info("GPU (CUDA) execution provider detected — using GPU acceleration")
            providers.append("CPUExecutionProvider")
            return providers

        except ImportError:
            logger.warning("onnxruntime not found, defaulting to CPU")
            return ["CPUExecutionProvider"]

    def detect_faces(self, image: np.ndarray, image_path: str) -> List[DetectedFace]:
        """
        Detect all faces in a single image.

        Args:
            image: BGR image array (from cv2.imread).
            image_path: Path to the source image (for metadata).

        Returns:
            List of DetectedFace objects for faces above the confidence threshold.
        """
        self.initialize()

        try:
            # InsightFace expects BGR images (which cv2 provides)
            faces = self.app.get(image)

            if not faces:
                logger.debug(f"No faces detected in {image_path}")
                return []

            # Sort by detection score (highest first) and limit
            faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
            if len(faces) > self.max_faces:
                logger.warning(
                    f"Capping faces from {len(faces)} to {self.max_faces} in {image_path}"
                )
                faces = faces[: self.max_faces]

            detected = []
            for idx, face in enumerate(faces):
                # Filter by detection threshold
                if face.det_score < self.det_thresh:
                    logger.debug(
                        f"Skipping face {idx} in {image_path}: "
                        f"det_score={face.det_score:.3f} < thresh={self.det_thresh}"
                    )
                    continue

                # Extract face crop from original image
                bbox = face.bbox.astype(int)
                x1 = max(0, bbox[0])
                y1 = max(0, bbox[1])
                x2 = min(image.shape[1], bbox[2])
                y2 = min(image.shape[0], bbox[3])
                face_crop = image[y1:y2, x1:x2].copy()

                detected_face = DetectedFace(
                    bbox=face.bbox,
                    landmarks=face.kps,  # 5-point keypoints (left_eye, right_eye, nose, left_mouth, right_mouth)
                    det_score=float(face.det_score),
                    face_index=idx,
                    image_path=str(image_path),
                    face_crop=face_crop,
                )
                detected.append(detected_face)

            logger.debug(f"Detected {len(detected)} faces in {image_path}")
            return detected

        except Exception as e:
            logger.error(f"Face detection failed for {image_path}: {e}")
            return []

    def detect_batch(
        self, images: List[Tuple[Path, np.ndarray]]
    ) -> Dict[str, List[DetectedFace]]:
        """
        Detect faces in a batch of images.

        Args:
            images: List of (path, image_array) tuples.

        Returns:
            Dict mapping image path (str) to list of detected faces.
        """
        self.initialize()
        results = {}
        total_faces = 0

        for path, image in images:
            faces = self.detect_faces(image, str(path))
            results[str(path)] = faces
            total_faces += len(faces)

        logger.info(f"Batch detection complete: {total_faces} faces in {len(images)} images")
        return results
