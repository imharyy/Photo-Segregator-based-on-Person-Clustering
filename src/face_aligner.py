"""
Face Aligner: Aligns detected faces using 5-point landmarks to a canonical 112x112 template.
This normalization step is critical for maximizing embedding quality.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("photo_segregator.face_aligner")

# Standard ArcFace / InsightFace alignment template (112x112)
# These are the target positions for: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],   # Left eye
        [73.5318, 51.5014],   # Right eye
        [56.0252, 71.7366],   # Nose tip
        [41.5493, 92.3655],   # Left mouth corner
        [70.7299, 92.2041],   # Right mouth corner
    ],
    dtype=np.float32,
)

# Output size for aligned face crops
ALIGNED_FACE_SIZE = (112, 112)


def estimate_affine_transform(
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray = ARCFACE_REFERENCE_LANDMARKS,
) -> np.ndarray:
    """
    Estimate a similarity transform (rotation, scale, translation) from
    source landmarks to destination template landmarks.

    Uses OpenCV's estimateAffinePartial2D for robustness.

    Args:
        src_landmarks: 5x2 array of detected landmark positions.
        dst_landmarks: 5x2 array of target positions (default: ArcFace template).

    Returns:
        2x3 affine transformation matrix.
    """
    # Ensure correct shape
    src = src_landmarks[:5].astype(np.float32).reshape(5, 2)
    dst = dst_landmarks[:5].astype(np.float32).reshape(5, 2)

    # estimateAffinePartial2D computes a similarity transform (4 DOF)
    # which is more appropriate than a full affine (6 DOF) for face alignment
    transform_matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.LMEDS
    )

    if transform_matrix is None:
        logger.warning("Affine transform estimation failed, using fallback")
        # Fallback: simple translation + scale based on eye positions
        transform_matrix = _fallback_transform(src, dst)

    return transform_matrix


def _fallback_transform(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """
    Compute a simple fallback transform when estimateAffinePartial2D fails.
    Uses only the two eye positions for a minimal similarity transform.
    """
    # Use eye centers for scale and rotation estimation
    src_eye_center = (src[0] + src[1]) / 2.0
    dst_eye_center = (dst[0] + dst[1]) / 2.0

    src_eye_dist = np.linalg.norm(src[1] - src[0])
    dst_eye_dist = np.linalg.norm(dst[1] - dst[0])

    if src_eye_dist < 1e-6:
        # Eyes too close together, return identity-like transform
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    scale = dst_eye_dist / src_eye_dist

    # Rotation angle between eye vectors
    src_vec = src[1] - src[0]
    dst_vec = dst[1] - dst[0]
    angle = np.arctan2(dst_vec[1], dst_vec[0]) - np.arctan2(src_vec[1], src_vec[0])

    cos_a = np.cos(angle) * scale
    sin_a = np.sin(angle) * scale

    tx = dst_eye_center[0] - cos_a * src_eye_center[0] + sin_a * src_eye_center[1]
    ty = dst_eye_center[1] - sin_a * src_eye_center[0] - cos_a * src_eye_center[1]

    return np.array([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=np.float32)


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = ALIGNED_FACE_SIZE,
) -> Optional[np.ndarray]:
    """
    Align a face by warping the image so that facial landmarks match the
    canonical ArcFace template.

    Args:
        image: Full BGR image containing the face.
        landmarks: Detected facial landmarks (at least 5 points, 5x2 array).
            Expected order: left_eye, right_eye, nose, left_mouth, right_mouth.
        output_size: Output image size (width, height). Default: (112, 112).

    Returns:
        Aligned face image (BGR, 112x112), or None if alignment fails.
    """
    try:
        # Handle different landmark formats
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 2)

        # If we have more than 5 landmarks (e.g., 106-point), extract the key 5
        if landmarks.shape[0] > 5:
            landmarks = _extract_5_from_106(landmarks)
        elif landmarks.shape[0] < 5:
            logger.warning(f"Insufficient landmarks ({landmarks.shape[0]}), cannot align")
            return None

        # Compute affine transform
        transform = estimate_affine_transform(landmarks)

        # Apply the warp
        aligned = cv2.warpAffine(
            image,
            transform,
            output_size,
            borderMode=cv2.BORDER_REPLICATE,
        )

        if aligned is None or aligned.size == 0:
            logger.warning("Alignment produced empty image")
            return None

        return aligned

    except Exception as e:
        logger.error(f"Face alignment failed: {e}")
        return None


def _extract_5_from_106(landmarks_106: np.ndarray) -> np.ndarray:
    """
    Extract 5 key landmarks from a 106-point landmark array.
    InsightFace 106-point layout mapping to 5 key points.
    """
    # Standard mapping for InsightFace 106-point landmarks:
    # Left eye center: average of points around the left eye (indices 33-42)
    # Right eye center: average of points around the right eye (indices 87-96)
    # Nose tip: index 86
    # Left mouth corner: index 52
    # Right mouth corner: index 61
    try:
        left_eye = np.mean(landmarks_106[33:43], axis=0)
        right_eye = np.mean(landmarks_106[87:97], axis=0)
        nose = landmarks_106[86]
        left_mouth = landmarks_106[52]
        right_mouth = landmarks_106[61]
        return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)
    except (IndexError, ValueError):
        # If 106-point mapping fails, just use first 5 points
        logger.warning("Failed to extract 5-point from 106-point landmarks, using first 5")
        return landmarks_106[:5].astype(np.float32)


def create_perturbation(
    image: np.ndarray,
    landmarks: np.ndarray,
    rotation_deg: float = 0.0,
    scale_factor: float = 1.0,
    output_size: Tuple[int, int] = ALIGNED_FACE_SIZE,
) -> Optional[np.ndarray]:
    """
    Create a perturbed version of an aligned face for ensemble embedding.
    Applies slight rotation and scale changes before alignment.

    Args:
        image: Full BGR image containing the face.
        landmarks: Detected facial landmarks (5x2).
        rotation_deg: Rotation in degrees to apply to landmarks.
        scale_factor: Scale factor to apply (1.0 = no change).
        output_size: Output image size.

    Returns:
        Perturbed aligned face image, or None if it fails.
    """
    try:
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 2)
        if landmarks.shape[0] > 5:
            landmarks = _extract_5_from_106(landmarks)

        # Apply perturbation to landmarks
        center = np.mean(landmarks[:5], axis=0)

        # Rotation
        angle_rad = np.deg2rad(rotation_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        perturbed = landmarks[:5].copy()
        perturbed -= center
        perturbed = perturbed @ rotation_matrix.T
        perturbed *= scale_factor
        perturbed += center

        # Compute transform with perturbed landmarks
        transform = estimate_affine_transform(perturbed.astype(np.float32))

        aligned = cv2.warpAffine(
            image,
            transform,
            output_size,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return aligned if aligned is not None and aligned.size > 0 else None

    except Exception as e:
        logger.debug(f"Perturbation creation failed: {e}")
        return None
