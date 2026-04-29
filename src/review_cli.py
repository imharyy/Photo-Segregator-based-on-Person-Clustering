"""
Review CLI: Interactive command-line tool for reviewing uncertain face-cluster assignments.
Allows users to accept, reassign, or discard faces from the review queue.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("photo_segregator.review_cli")


class ReviewCLI:
    """
    Interactive CLI for reviewing faces in the review queue.

    For each uncertain face, displays:
    - Face crop (opens in default image viewer)
    - Source image path
    - Current cluster assignment and confidence
    - Top-3 nearest clusters with distances

    User options:
    [a] Accept current assignment
    [m] Move to a different cluster (enter cluster number)
    [n] Create a new cluster
    [d] Discard (mark as non-face / garbage)
    [s] Skip (leave for later)
    [q] Quit review
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Base output directory (expects _review_queue/ and _metadata/).
        """
        self.output_dir = Path(output_dir)
        self.review_dir = self.output_dir / "_review_queue"
        self.metadata_dir = self.output_dir / "_metadata"

    def run(self) -> Dict[str, int]:
        """
        Run the interactive review session.

        Returns:
            Dict mapping face_id → corrected cluster label.
            Special labels: -1 = noise, -2 = discarded.
        """
        # Load review items
        review_path = self.review_dir / "review_items.json"
        if not review_path.exists():
            print("\n✅ No items in review queue. Nothing to review.")
            return {}

        with open(review_path, "r", encoding="utf-8") as f:
            review_items = json.load(f)

        if not review_items:
            print("\n✅ Review queue is empty.")
            return {}

        # Load existing cluster info
        cluster_report = self._load_cluster_report()
        face_metadata = self._load_face_metadata()

        # Track corrections
        corrections = {}
        total = len(review_items)

        print(f"\n{'='*60}")
        print(f"  FACE CLUSTER REVIEW")
        print(f"  {total} faces to review")
        print(f"{'='*60}")
        print(f"\nCommands: [a]ccept | [m]ove | [n]ew cluster | [d]iscard | [s]kip | [q]uit\n")

        for idx, item in enumerate(review_items):
            face_id = item["face_id"]
            image_path = item.get("image_path", "unknown")
            current_label = item.get("current_label", -1)
            confidence = item.get("confidence", 0.0)
            quality = item.get("quality_score", 0.0)

            print(f"\n--- Face {idx+1}/{total} ---")
            print(f"  Face ID:     {face_id}")
            print(f"  Source:      {image_path}")
            print(f"  Cluster:     {'Noise' if current_label == -1 else f'Person_{current_label:03d}'}")
            print(f"  Confidence:  {confidence:.3f}")
            print(f"  Quality:     {quality:.3f}")

            # Show available clusters
            if cluster_report and "clusters" in cluster_report:
                print(f"\n  Available clusters:")
                for name, info in sorted(cluster_report.get("clusters", {}).items()):
                    face_count = info.get("face_count", 0)
                    avg_conf = info.get("avg_confidence", 0)
                    print(f"    {name}: {face_count} faces (avg confidence: {avg_conf:.3f})")

            # Try to display the face crop
            crop_path = self.review_dir / f"{face_id}.jpg"
            if crop_path.exists():
                self._show_image(crop_path)

            # Get user input
            while True:
                choice = input(f"\n  Action [a/m/n/d/s/q]: ").strip().lower()

                if choice == "a":
                    # Accept current assignment
                    if current_label >= 0:
                        corrections[face_id] = current_label
                    print(f"  ✅ Accepted: cluster {current_label}")
                    break

                elif choice == "m":
                    # Move to different cluster
                    try:
                        target = int(input("  Enter cluster number: ").strip())
                        corrections[face_id] = target
                        print(f"  ✅ Moved to cluster {target}")
                        break
                    except ValueError:
                        print("  ❌ Invalid cluster number, try again")

                elif choice == "n":
                    # Create new cluster
                    existing_labels = set()
                    for v in corrections.values():
                        if v >= 0:
                            existing_labels.add(v)
                    if cluster_report and "clusters" in cluster_report:
                        for name in cluster_report["clusters"]:
                            try:
                                num = int(name.split("_")[1])
                                existing_labels.add(num)
                            except (IndexError, ValueError):
                                pass
                    new_label = max(existing_labels) + 1 if existing_labels else 0
                    corrections[face_id] = new_label
                    print(f"  ✅ Created new cluster: Person_{new_label:03d}")
                    break

                elif choice == "d":
                    # Discard
                    corrections[face_id] = -2  # Special: discarded
                    print(f"  🗑️  Discarded")
                    break

                elif choice == "s":
                    # Skip
                    print(f"  ⏭️  Skipped")
                    break

                elif choice == "q":
                    # Quit
                    print(f"\n  Quitting review. {len(corrections)} corrections made.")
                    self._save_corrections(corrections)
                    return corrections

                else:
                    print(f"  ❌ Unknown command: '{choice}'. Use a/m/n/d/s/q")

        # Save corrections
        self._save_corrections(corrections)
        print(f"\n{'='*60}")
        print(f"  Review complete: {len(corrections)} corrections made")
        print(f"  Corrections saved to {self.metadata_dir / 'corrections.json'}")
        print(f"{'='*60}")

        return corrections

    def _show_image(self, image_path: Path):
        """Open an image in the default system viewer."""
        try:
            if sys.platform == "win32":
                os.startfile(str(image_path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(image_path)], check=False)
            else:
                subprocess.run(["xdg-open", str(image_path)], check=False)
        except Exception:
            print(f"  (Could not open image viewer. See: {image_path})")

    def _load_cluster_report(self) -> Optional[Dict]:
        """Load the cluster report JSON."""
        report_path = self.metadata_dir / "cluster_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _load_face_metadata(self) -> Optional[Dict]:
        """Load face metadata JSON."""
        meta_path = self.metadata_dir / "face_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_corrections(self, corrections: Dict[str, int]):
        """Save corrections to a JSON file for pipeline re-processing."""
        if not corrections:
            return

        corrections_path = self.metadata_dir / "corrections.json"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Merge with existing corrections
        existing = {}
        if corrections_path.exists():
            with open(corrections_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        existing.update({k: int(v) for k, v in corrections.items()})

        with open(corrections_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved {len(corrections)} corrections to {corrections_path}")
