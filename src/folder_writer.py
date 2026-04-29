"""
Folder Writer: Creates person folders, copies images and face crops into them.
Handles multi-face images (same photo in multiple person folders).
Generates comprehensive metadata files.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .utils import ensure_dir

logger = logging.getLogger("photo_segregator.folder_writer")


class FolderWriter:
    """
    Creates the output directory structure:
    - Person_001/ — original images + crops/ subfolder
    - Person_002/ — etc.
    - _review_queue/ — uncertain faces for manual review
    - _metadata/ — face_metadata.json, cluster_report.json
    """

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Args:
            config: Configuration dict.
            output_dir: Base output directory path.
        """
        self.output_dir = Path(output_dir)
        self.save_crops = config.get("save_crops", True)
        self.copy_mode = config.get("copy_mode", "copy")

    def write_clusters(
        self,
        labels: np.ndarray,
        confidences: np.ndarray,
        face_ids: List[str],
        face_metadata: Dict[str, Dict],
        aligned_faces: Dict[str, np.ndarray],
        review_queue: List[str],
    ) -> Dict[str, Any]:
        """
        Create person folders and copy/link images into them.

        Args:
            labels: Cluster labels per face.
            confidences: Confidence scores per face.
            face_ids: Face IDs.
            face_metadata: Dict mapping face_id → metadata dict.
            aligned_faces: Dict mapping face_id → aligned face crop (BGR array).
            review_queue: List of face IDs flagged for review.

        Returns:
            Summary dict with statistics.
        """
        logger.info(f"Writing output to {self.output_dir}")

        # Clean output directories (but preserve _metadata)
        self._clean_output()

        unique_labels = sorted(set(labels) - {-1})
        review_set = set(review_queue)

        # Track statistics
        stats = {
            "n_clusters": len(unique_labels),
            "n_faces": len(face_ids),
            "n_review": len(review_queue),
            "n_noise": int(np.sum(labels == -1)),
            "clusters": {},
        }

        # Create person folders
        for label in unique_labels:
            person_dir = self.output_dir / f"Person_{label:03d}"
            ensure_dir(person_dir)

            if self.save_crops:
                ensure_dir(person_dir / "crops")

            # Find all faces in this cluster
            cluster_face_ids = [
                fid for fid, l in zip(face_ids, labels) if l == label
            ]

            # Track unique images in this cluster
            images_copied = set()
            cluster_info = {
                "face_count": len(cluster_face_ids),
                "images": [],
                "avg_confidence": 0.0,
            }

            total_conf = 0.0
            for fid in cluster_face_ids:
                meta = face_metadata.get(fid, {})
                confidence = confidences[face_ids.index(fid)]
                total_conf += confidence

                image_path = meta.get("image_path", "")
                if not image_path:
                    continue

                src = Path(image_path)
                if not src.exists():
                    logger.warning(f"Source image not found: {image_path}")
                    continue

                # Copy original image (if not already copied for this cluster)
                if str(src) not in images_copied:
                    dst = person_dir / src.name

                    # Handle name collisions
                    if dst.exists():
                        stem = dst.stem
                        suffix = dst.suffix
                        counter = 1
                        while dst.exists():
                            dst = person_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    try:
                        shutil.copy2(str(src), str(dst))
                        images_copied.add(str(src))
                        cluster_info["images"].append({
                            "source": str(src),
                            "destination": str(dst),
                            "face_id": fid,
                            "confidence": float(confidence),
                        })
                    except Exception as e:
                        logger.error(f"Failed to copy {src} → {dst}: {e}")

                # Save aligned face crop
                if self.save_crops and fid in aligned_faces:
                    crop = aligned_faces[fid]
                    if crop is not None:
                        crop_path = person_dir / "crops" / f"{fid}.jpg"
                        try:
                            cv2.imwrite(str(crop_path), crop)
                        except Exception as e:
                            logger.error(f"Failed to save crop {crop_path}: {e}")

            cluster_info["avg_confidence"] = total_conf / max(len(cluster_face_ids), 1)
            stats["clusters"][f"Person_{label:03d}"] = cluster_info

        # Write review queue
        review_dir = ensure_dir(self.output_dir / "_review_queue")
        review_items = []

        for fid in review_queue:
            meta = face_metadata.get(fid, {})
            label = int(labels[face_ids.index(fid)]) if fid in face_ids else -1
            confidence = float(confidences[face_ids.index(fid)]) if fid in face_ids else 0.0

            # Save face crop to review queue
            if fid in aligned_faces and aligned_faces[fid] is not None:
                crop_path = review_dir / f"{fid}.jpg"
                try:
                    cv2.imwrite(str(crop_path), aligned_faces[fid])
                except Exception:
                    pass

            review_items.append({
                "face_id": fid,
                "image_path": meta.get("image_path", ""),
                "current_label": label,
                "confidence": confidence,
                "quality_score": meta.get("quality_score", 0),
            })

        # Save review queue metadata
        review_meta_path = review_dir / "review_items.json"
        with open(review_meta_path, "w", encoding="utf-8") as f:
            json.dump(review_items, f, indent=2, default=str)

        # Save cluster report
        self._save_cluster_report(stats, face_ids, labels, confidences, face_metadata)

        logger.info(
            f"Output written: {len(unique_labels)} person folders, "
            f"{len(review_queue)} review items"
        )
        return stats

    def _clean_output(self):
        """Remove existing person folders (but preserve _metadata and _logs)."""
        if not self.output_dir.exists():
            ensure_dir(self.output_dir)
            return

        for item in self.output_dir.iterdir():
            if item.is_dir() and item.name.startswith("Person_"):
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_dir() and item.name == "_review_queue":
                shutil.rmtree(item, ignore_errors=True)

    def _save_cluster_report(
        self,
        stats: Dict,
        face_ids: List[str],
        labels: np.ndarray,
        confidences: np.ndarray,
        face_metadata: Dict[str, Dict],
    ):
        """Save comprehensive cluster report as JSON."""
        metadata_dir = ensure_dir(self.output_dir / "_metadata")

        # Full face-to-cluster mapping
        face_mapping = {}
        for fid, label, conf in zip(face_ids, labels, confidences):
            meta = face_metadata.get(fid, {})
            face_mapping[fid] = {
                "cluster_label": int(label),
                "cluster_name": f"Person_{label:03d}" if label >= 0 else "noise",
                "confidence": float(conf),
                "image_path": meta.get("image_path", ""),
                "bbox": meta.get("bbox", []),
                "quality_score": meta.get("quality_score", 0),
                "det_score": meta.get("det_score", 0),
            }

        # Save face metadata
        face_meta_path = metadata_dir / "face_metadata.json"
        with open(face_meta_path, "w", encoding="utf-8") as f:
            json.dump(face_mapping, f, indent=2, default=str)

        # Save cluster report
        report_path = metadata_dir / "cluster_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

        logger.info(f"Cluster report saved to {metadata_dir}")
