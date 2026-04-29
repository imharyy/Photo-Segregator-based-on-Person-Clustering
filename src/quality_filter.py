"""
Quality Filter: Evaluates detected faces for blur, size, pose, and occlusion.
Returns a composite quality score and pass/fail decision with reasons.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("photo_segregator.quality_filter")


@dataclass
class QualityResult:
    """Result of quality assessment for a single face."""
    passed: bool                 # Whether the face passes quality checks
    composite_score: float       # Overall quality score (0-1, higher = better)
    scores: Dict[str, float]    # Individual quality scores
    reasons: List[str]          # Reasons for failure (empty if passed)

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"QualityResult({status}, score={self.composite_score:.3f}, reasons={self.reasons})"


class QualityFilter:
    """
    Multi-criteria quality filter for detected faces.
    Evaluates size, blur, pose, and occlusion indicators.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize quality filter with configuration.

        Args:
            config: Configuration dict with 'quality' section.
        """
        quality_cfg = config.get("quality", {})
        self.min_face_size = quality_cfg.get("min_face_size", 40)
        self.blur_threshold = quality_cfg.get("blur_threshold", 50.0)
        self.max_pose_angle = quality_cfg.get("max_pose_angle", 75.0)
        self.min_landmark_conf = quality_cfg.get("min_landmark_conf", 0.3)

    def evaluate(
        self,
        face_crop: np.ndarray,
        bbox: np.ndarray,
        landmarks: np.ndarray,
        det_score: float,
    ) -> QualityResult:
        """
        Evaluate the quality of a detected face.

        Args:
            face_crop: Cropped face image (BGR).
            bbox: Face bounding box [x1, y1, x2, y2].
            landmarks: 5x2 facial landmarks.
            det_score: Detection confidence score.

        Returns:
            QualityResult with pass/fail, scores, and reasons.
        """
        scores = {}
        reasons = []

        # 1. Size check
        size_score = self._check_size(bbox)
        scores["size"] = size_score
        if size_score < 0.5:
            reasons.append(f"Face too small: {min(bbox[2]-bbox[0], bbox[3]-bbox[1]):.0f}px < {self.min_face_size}px")

        # 2. Blur check
        blur_score = self._check_blur(face_crop)
        scores["blur"] = blur_score
        if blur_score < 0.3:
            reasons.append(f"Face too blurry (Laplacian variance low)")

        # 3. Pose check (using landmarks geometry)
        pose_score = self._check_pose(landmarks)
        scores["pose"] = pose_score
        if pose_score < 0.3:
            reasons.append(f"Extreme face pose detected")

        # 4. Detection confidence
        conf_score = min(det_score, 1.0)
        scores["detection_confidence"] = conf_score
        if conf_score < self.min_landmark_conf:
            reasons.append(f"Low detection confidence: {conf_score:.3f}")

        # 5. Aspect ratio check (catches false detections)
        aspect_score = self._check_aspect_ratio(bbox)
        scores["aspect_ratio"] = aspect_score
        if aspect_score < 0.3:
            reasons.append(f"Unusual face aspect ratio")

        # Composite score: weighted average
        weights = {
            "size": 0.25,
            "blur": 0.25,
            "pose": 0.20,
            "detection_confidence": 0.20,
            "aspect_ratio": 0.10,
        }
        composite = sum(scores[k] * weights[k] for k in weights)
        passed = len(reasons) == 0

        result = QualityResult(
            passed=passed,
            composite_score=composite,
            scores=scores,
            reasons=reasons,
        )

        if not passed:
            logger.debug(f"Face quality check FAILED: {reasons}")

        return result

    def _check_size(self, bbox: np.ndarray) -> float:
        """
        Check if the face meets minimum size requirements.
        Returns a score from 0 to 1.
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        min_side = min(width, height)

        if min_side < self.min_face_size * 0.5:
            return 0.0
        elif min_side < self.min_face_size:
            return 0.3
        elif min_side < self.min_face_size * 2:
            return 0.7
        else:
            return 1.0

    def _check_blur(self, face_crop: np.ndarray) -> float:
        """
        Detect blur using Laplacian variance.
        Higher variance = sharper image = higher score.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0

        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize: map laplacian variance to 0-1 score
            if laplacian_var < self.blur_threshold * 0.3:
                return 0.1
            elif laplacian_var < self.blur_threshold:
                return 0.5
            elif laplacian_var < self.blur_threshold * 3:
                return 0.8
            else:
                return 1.0
        except Exception:
            return 0.5  # Unknown quality

    def _check_pose(self, landmarks: np.ndarray) -> float:
        """
        Estimate face pose from landmark geometry.
        Uses the ratio of nose-to-eye distances to detect side faces.
        Returns a score from 0 (extreme pose) to 1 (frontal).
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.5  # Unknown pose

        try:
            # Reshape if needed
            if landmarks.ndim == 1:
                landmarks = landmarks.reshape(-1, 2)

            # Use first 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            lm = landmarks[:5]
            left_eye, right_eye, nose = lm[0], lm[1], lm[2]

            # Inter-eye distance
            eye_dist = np.linalg.norm(right_eye - left_eye)
            if eye_dist < 1e-6:
                return 0.1  # Eyes overlapping = very extreme pose

            # Nose offset from eye midpoint (horizontal component)
            eye_mid = (left_eye + right_eye) / 2.0
            nose_offset_x = abs(nose[0] - eye_mid[0])

            # Ratio: nose horizontal offset / inter-eye distance
            # Frontal: ~0, Side face: > 0.3
            pose_ratio = nose_offset_x / eye_dist

            # Inter-eye angle (head tilt)
            eye_vec = right_eye - left_eye
            tilt_angle = abs(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))

            # Combine: pose_ratio for yaw, tilt_angle for roll
            if pose_ratio > 0.5 or tilt_angle > self.max_pose_angle:
                return 0.1
            elif pose_ratio > 0.3 or tilt_angle > 45:
                return 0.4
            elif pose_ratio > 0.15 or tilt_angle > 20:
                return 0.7
            else:
                return 1.0

        except Exception:
            return 0.5  # Unknown pose

    def _check_aspect_ratio(self, bbox: np.ndarray) -> float:
        """
        Check that the face bounding box has a reasonable aspect ratio.
        Real faces are roughly square (0.6 to 1.4 ratio).
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        if width < 1 or height < 1:
            return 0.0

        ratio = width / height
        if 0.6 <= ratio <= 1.4:
            return 1.0
        elif 0.4 <= ratio <= 1.8:
            return 0.6
        else:
            return 0.2  # Suspicious aspect ratio

    def get_confidence_penalty(self, quality_result: QualityResult) -> float:
        """
        Compute a confidence penalty factor based on quality issues.
        Used downstream to adjust cluster assignment confidence.

        Returns:
            Penalty factor (0-1). 1.0 = no penalty, 0.0 = maximum penalty.
        """
        if quality_result.passed:
            return 1.0

        # Penalty proportional to number and severity of issues
        penalty = quality_result.composite_score
        return max(0.2, penalty)  # Never penalize below 0.2 (face was still detected)
