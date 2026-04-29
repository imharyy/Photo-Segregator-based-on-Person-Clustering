"""
Photo Segregator — Main Pipeline Entry Point

Automatically groups photos by person using:
- RetinaFace face detection + landmark extraction
- 5-point alignment to canonical ArcFace template
- ArcFace 512-d embeddings with ensemble averaging
- UMAP dimensionality reduction
- HDBSCAN density-based clustering (DBSCAN fallback)
- Adaptive thresholding + cluster refinement
- Semi-supervised learning from review corrections
- Confidence heatmap visualization

Usage:
    python main.py --input ./photos --output ./output
    python main.py --input ./photos --output ./output --incremental
    python main.py --review --output ./output
    python main.py --evaluate --output ./output --ground-truth ./labels.json
    python main.py --calibrate --input ./photos --ground-truth ./labels.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import load_config, setup_logging, compute_image_hash, format_face_id, ensure_dir
from src.image_loader import discover_images, load_image
from src.face_detector import FaceDetector, DetectedFace
from src.face_aligner import align_face
from src.quality_filter import QualityFilter
from src.embedding_extractor import EmbeddingExtractor
from src.ensemble_embeddings import EnsembleEmbedder
from src.embedding_cache import EmbeddingCache
from src.dimensionality_reduction import DimensionalityReducer
from src.adaptive_threshold import AdaptiveThreshold
from src.clustering import ClusteringEngine
from src.cluster_refinement import ClusterRefiner
from src.semi_supervised import SemiSupervisedLearner
from src.confidence_heatmap import ConfidenceHeatmap
from src.folder_writer import FolderWriter
from src.review_cli import ReviewCLI
from src.evaluation import Evaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Photo Segregator — Automatically group photos by person",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input folder containing images",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Path to output folder (default: ./output)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process new images (skip already-cached ones)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Launch interactive review CLI for uncertain faces",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate clustering against ground-truth labels",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground-truth labels JSON (for --evaluate)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate clustering parameters on labeled data",
    )
    return parser.parse_args()


def run_pipeline(
    input_dir: str,
    output_dir: str,
    config: dict,
    incremental: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    Run the full photo segregation pipeline.

    Steps:
    1. Discover and load images
    2. Detect faces + extract landmarks
    3. Align faces
    4. Filter quality
    5. Extract ensemble embeddings
    6. Cache embeddings
    7. Semi-supervised classification of known identities
    8. UMAP dimensionality reduction
    9. Adaptive threshold computation
    10. HDBSCAN clustering
    11. Cluster refinement
    12. Confidence heatmap generation
    13. Folder writing
    """
    if logger is None:
        logger = logging.getLogger("photo_segregator")

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("  PHOTO SEGREGATOR PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Mode:   {'incremental' if incremental else 'full'}")

    # ---- Initialize components ----
    logger.info("\n[1/13] Initializing components...")
    detector = FaceDetector(config)
    quality_filter = QualityFilter(config)
    extractor = EmbeddingExtractor(config)
    ensemble = EnsembleEmbedder(config, extractor)
    cache = EmbeddingCache(output_dir)
    dim_reducer = DimensionalityReducer(config)
    adaptive_thresh = AdaptiveThreshold(config)
    clustering_engine = ClusteringEngine(config)
    semi_supervised = SemiSupervisedLearner(config, output_dir)
    heatmap = ConfidenceHeatmap(config, output_dir)
    writer = FolderWriter(config, output_dir)

    # Load existing cache for incremental mode
    if incremental:
        cache.load()
        semi_supervised.load()
        logger.info(f"Loaded {cache.count} existing embeddings from cache")

    # ---- Step 1: Discover images ----
    logger.info("\n[2/13] Discovering images...")
    image_paths = discover_images(
        input_dir,
        recursive=config.get("recursive", True),
    )

    if not image_paths:
        logger.warning("No images found in input directory!")
        return

    # In incremental mode, skip already-processed images
    if incremental and cache.count > 0:
        processed_hashes = cache.get_processed_hashes()
        new_paths = []
        for path in image_paths:
            try:
                img_hash = compute_image_hash(str(path))
                if img_hash not in processed_hashes:
                    new_paths.append(path)
            except Exception:
                new_paths.append(path)  # Process if we can't hash

        logger.info(f"Incremental: {len(new_paths)} new images to process ({len(image_paths) - len(new_paths)} skipped)")
        image_paths = new_paths

        if not image_paths and cache.count > 0:
            logger.info("No new images. Re-clustering existing embeddings...")
            # Jump to clustering with existing cache
            embeddings, face_ids = cache.get_all_embeddings()
            aligned_faces_dict = {}  # No new crops
            # Skip to step 8
            return _run_clustering_phase(
                embeddings, face_ids, cache, dim_reducer, adaptive_thresh,
                clustering_engine, semi_supervised, heatmap, writer, config,
                output_dir, aligned_faces_dict, logger,
            )

    # ---- Step 2-6: Process each image ----
    logger.info(f"\n[3/13] Processing {len(image_paths)} images (detect → align → quality → embed)...")

    all_face_ids = []
    aligned_faces_dict = {}  # face_id → aligned crop
    faces_processed = 0
    faces_skipped = 0

    for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
        # Load image
        image = load_image(img_path)
        if image is None:
            continue

        # Compute image hash
        try:
            img_hash = compute_image_hash(str(img_path))
        except Exception as e:
            logger.error(f"Failed to hash {img_path}: {e}")
            continue

        # Detect faces
        detected_faces = detector.detect_faces(image, str(img_path))

        if not detected_faces:
            logger.debug(f"No faces in {img_path.name}")
            continue

        for face in detected_faces:
            face_id = format_face_id(img_hash, face.face_index)

            # Quality check
            quality_result = quality_filter.evaluate(
                face.face_crop, face.bbox, face.landmarks, face.det_score
            )

            # Skip hard failures (too small, etc.) but keep soft failures with reduced confidence
            if not quality_result.passed and quality_result.composite_score < 0.2:
                faces_skipped += 1
                logger.debug(f"Skipped {face_id}: {quality_result.reasons}")
                continue

            # Align face
            aligned = align_face(image, face.landmarks)
            if aligned is None:
                faces_skipped += 1
                logger.debug(f"Alignment failed for {face_id}")
                continue

            # Extract ensemble embedding
            embedding = ensemble.extract_ensemble(
                aligned,
                original_image=image,
                landmarks=face.landmarks,
            )
            if embedding is None:
                faces_skipped += 1
                logger.debug(f"Embedding failed for {face_id}")
                continue

            # Store results
            face_meta = {
                "image_path": str(img_path.resolve()),
                "image_hash": img_hash,
                "face_index": face.face_index,
                "bbox": face.bbox.tolist(),
                "landmarks": face.landmarks.tolist() if face.landmarks is not None else [],
                "det_score": float(face.det_score),
                "quality_score": float(quality_result.composite_score),
                "quality_passed": quality_result.passed,
                "quality_reasons": quality_result.reasons,
            }

            cache.add_face(face_id, embedding, face_meta)
            aligned_faces_dict[face_id] = aligned
            all_face_ids.append(face_id)
            faces_processed += 1

    logger.info(f"Processed {faces_processed} faces ({faces_skipped} skipped)")

    # Save cache
    logger.info("\n[6/13] Saving embedding cache...")
    cache.save()

    if cache.count == 0:
        logger.warning("No faces detected in any images. Pipeline complete.")
        return

    # ---- Clustering phase ----
    embeddings, face_ids = cache.get_all_embeddings()
    _run_clustering_phase(
        embeddings, face_ids, cache, dim_reducer, adaptive_thresh,
        clustering_engine, semi_supervised, heatmap, writer, config,
        output_dir, aligned_faces_dict, logger,
    )

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"  Pipeline complete in {elapsed:.1f}s")
    logger.info(f"{'='*60}")


def _run_clustering_phase(
    embeddings, face_ids, cache, dim_reducer, adaptive_thresh,
    clustering_engine, semi_supervised, heatmap, writer, config,
    output_dir, aligned_faces_dict, logger,
):
    """Run the clustering, refinement, and output phase."""

    # ---- Step 7: Semi-supervised classification ----
    logger.info("\n[7/13] Semi-supervised classification...")
    known_labels, unknown_indices = semi_supervised.classify_known(embeddings, face_ids)

    # ---- Step 8: UMAP dimensionality reduction ----
    logger.info("\n[8/13] UMAP dimensionality reduction...")
    reduced_embeddings = dim_reducer.fit_transform(embeddings)

    # ---- Step 9: Adaptive thresholds ----
    logger.info("\n[9/13] Computing adaptive thresholds...")
    thresholds = adaptive_thresh.compute(embeddings)

    # ---- Step 10: Clustering ----
    logger.info("\n[10/13] Clustering with HDBSCAN...")

    # If we have known labels from semi-supervised, we cluster only unknowns
    # and assign known ones directly
    if known_labels and len(unknown_indices) < len(face_ids):
        # Cluster only unknown faces
        unknown_embeddings = reduced_embeddings[unknown_indices]
        if len(unknown_embeddings) > 1:
            unknown_labels, unknown_probs = clustering_engine.cluster(unknown_embeddings)
        elif len(unknown_embeddings) == 1:
            unknown_labels = np.array([0])
            unknown_probs = np.array([0.5])
        else:
            unknown_labels = np.array([])
            unknown_probs = np.array([])

        # Merge known and unknown labels
        # Offset unknown cluster labels to avoid collision with known identity labels
        max_known = max(known_labels.values()) if known_labels else -1
        if len(unknown_labels) > 0:
            unknown_labels[unknown_labels >= 0] += max_known + 1

        # Build final labels and probabilities arrays
        labels = np.full(len(face_ids), -1, dtype=int)
        probabilities = np.full(len(face_ids), 0.0, dtype=float)

        for fid, label in known_labels.items():
            idx = face_ids.index(fid)
            labels[idx] = label
            probabilities[idx] = 0.95  # High confidence for known

        for i, orig_idx in enumerate(unknown_indices):
            if i < len(unknown_labels):
                labels[orig_idx] = int(unknown_labels[i])
                probabilities[orig_idx] = float(unknown_probs[i])

        logger.info(f"Combined: {len(known_labels)} known + {len(unknown_indices)} clustered")
    else:
        # Full clustering (no known identities)
        labels, probabilities = clustering_engine.cluster(reduced_embeddings)

    # Log clustering summary
    summary = clustering_engine.get_cluster_summary()
    logger.info(f"Clustering: {summary.get('n_clusters', 0)} clusters, {summary.get('n_noise', 0)} noise")

    # ---- Step 11: Cluster refinement ----
    logger.info("\n[11/13] Refining clusters...")
    refiner = ClusterRefiner(config, adaptive_thresholds=thresholds)
    labels, confidences, review_queue = refiner.refine(
        embeddings, labels, probabilities, face_ids,
    )

    # Update cache with cluster labels
    label_dict = {fid: int(l) for fid, l in zip(face_ids, labels)}
    conf_dict = {fid: float(c) for fid, c in zip(face_ids, confidences)}
    cache.update_cluster_labels(label_dict, conf_dict)
    cache.save()

    # ---- Step 12: Confidence heatmap ----
    logger.info("\n[12/13] Generating confidence heatmap...")
    heatmap.generate(
        embeddings, labels, confidences, face_ids, review_queue,
        dim_reducer=dim_reducer,
    )

    # ---- Step 13: Write output folders ----
    logger.info("\n[13/13] Writing output folders...")
    stats = writer.write_clusters(
        labels, confidences, face_ids,
        cache.metadata, aligned_faces_dict, review_queue,
    )

    # Save adaptive threshold report
    threshold_report = adaptive_thresh.get_report()
    report_path = Path(output_dir) / "_metadata" / "threshold_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(threshold_report, f, indent=2, default=str)

    logger.info(f"\nResults: {stats.get('n_clusters', 0)} person folders, "
                f"{stats.get('n_review', 0)} review items, "
                f"{stats.get('n_noise', 0)} unassigned")


def run_review(output_dir: str):
    """Launch the interactive review CLI."""
    cli = ReviewCLI(output_dir)
    corrections = cli.run()

    if corrections:
        # Learn from corrections
        config = load_config()
        cache = EmbeddingCache(output_dir)
        cache.load()

        embeddings, face_ids = cache.get_all_embeddings()
        if len(face_ids) > 0:
            learner = SemiSupervisedLearner(config, output_dir)
            learner.learn_from_corrections(embeddings, face_ids, corrections)
            print("\n💡 Run the pipeline again to apply learned corrections:")
            print(f"   python main.py --input <photos> --output {output_dir}")


def run_evaluation(output_dir: str, ground_truth_path: str):
    """Run evaluation against ground-truth labels."""
    evaluator = Evaluator(output_dir)

    # Load ground truth
    gt_labels = evaluator.load_ground_truth(ground_truth_path)

    # Load predicted labels from metadata
    cache = EmbeddingCache(output_dir)
    cache.load()

    if cache.count == 0:
        print("❌ No cached embeddings found. Run the pipeline first.")
        return

    # Match face IDs between predictions and ground truth
    _, face_ids = cache.get_all_embeddings()

    predicted = []
    truth = []

    for fid in face_ids:
        meta = cache.get_metadata(fid)
        if meta is None:
            continue

        # Try to match by face_id or image_path
        gt_label = gt_labels.get(fid)
        if gt_label is None:
            img_path = meta.get("image_path", "")
            gt_label = gt_labels.get(img_path)
            if gt_label is None:
                gt_label = gt_labels.get(Path(img_path).name)

        if gt_label is not None:
            predicted.append(meta.get("cluster_label", -1))
            truth.append(gt_label)

    if len(predicted) < 2:
        print(f"❌ Only {len(predicted)} faces matched to ground truth. Need at least 2.")
        return

    results = evaluator.evaluate(
        np.array(predicted),
        np.array(truth),
        face_ids,
    )


def run_calibration(input_dir: str, ground_truth_path: str, config: dict):
    """
    Calibrate clustering parameters by grid search on labeled data.
    Tests combinations of min_cluster_size and min_samples.
    """
    print("\n🔧 Running parameter calibration...")

    # Load ground truth
    gt_labels = Evaluator.load_ground_truth(ground_truth_path)

    # First, run detection + embedding with current config
    output_dir = "./calibration_temp"
    run_pipeline(input_dir, output_dir, config)

    cache = EmbeddingCache(output_dir)
    cache.load()
    embeddings, face_ids = cache.get_all_embeddings()

    if len(face_ids) == 0:
        print("❌ No faces detected. Cannot calibrate.")
        return

    # Build ground truth array
    gt_array = []
    valid_indices = []
    for i, fid in enumerate(face_ids):
        meta = cache.get_metadata(fid)
        gt = gt_labels.get(fid) or gt_labels.get(meta.get("image_path", ""))
        if gt is not None:
            gt_array.append(gt)
            valid_indices.append(i)

    if len(gt_array) < 4:
        print("❌ Too few labeled samples for calibration")
        return

    gt_array = np.array(gt_array)
    valid_embeddings = embeddings[valid_indices]

    # UMAP reduction
    reducer = DimensionalityReducer(config)
    reduced = reducer.fit_transform(valid_embeddings)

    # Grid search
    param_grid = {
        "min_cluster_size": [2, 3, 5],
        "min_samples": [1, 2, 3],
    }

    best_f1 = -1
    best_params = {}
    results = []

    for mcs in param_grid["min_cluster_size"]:
        for ms in param_grid["min_samples"]:
            try:
                from sklearn.cluster import HDBSCAN
                clusterer = HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    metric="euclidean",
                    cluster_selection_method="eom",
                )
                pred_labels = clusterer.fit_predict(reduced)

                valid_mask = pred_labels >= 0
                if np.sum(valid_mask) < 2:
                    continue

                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(gt_array[valid_mask], pred_labels[valid_mask])

                # Compute pairwise F1
                evaluator = Evaluator(output_dir)
                metrics = evaluator._compute_pairwise_metrics(
                    pred_labels[valid_mask], gt_array[valid_mask]
                )
                f1 = metrics["pairwise_f1"]
                noise_ratio = np.sum(~valid_mask) / len(pred_labels)

                result = {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "f1": f1,
                    "ari": ari,
                    "noise_ratio": noise_ratio,
                    "n_clusters": len(set(pred_labels) - {-1}),
                }
                results.append(result)

                print(f"  mcs={mcs}, ms={ms}: F1={f1:.4f}, ARI={ari:.4f}, "
                      f"clusters={result['n_clusters']}, noise={noise_ratio:.1%}")

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {"min_cluster_size": mcs, "min_samples": ms}

            except Exception as e:
                print(f"  mcs={mcs}, ms={ms}: ERROR - {e}")

    print(f"\n✅ Best parameters: {best_params} (F1={best_f1:.4f})")
    print(f"   Update config.yaml → clustering.hdbscan with these values.")

    # Save calibration results
    cal_path = Path(output_dir) / "_metadata" / "calibration_results.json"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cal_path, "w") as f:
        json.dump({"best_params": best_params, "best_f1": best_f1, "all_results": results}, f, indent=2)


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    output_dir = args.output or config.get("output_dir", "./output")

    # Setup logging
    logger = setup_logging(config, output_dir)

    # Route to the appropriate mode
    if args.review:
        run_review(output_dir)

    elif args.evaluate:
        if not args.ground_truth:
            print("❌ --ground-truth is required for evaluation mode")
            sys.exit(1)
        run_evaluation(output_dir, args.ground_truth)

    elif args.calibrate:
        if not args.input or not args.ground_truth:
            print("❌ --input and --ground-truth are required for calibration mode")
            sys.exit(1)
        run_calibration(args.input, args.ground_truth, config)

    else:
        # Full pipeline
        if not args.input:
            input_dir = config.get("input_dir", "./photos")
        else:
            input_dir = args.input

        run_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            incremental=args.incremental,
            logger=logger,
        )


if __name__ == "__main__":
    main()
