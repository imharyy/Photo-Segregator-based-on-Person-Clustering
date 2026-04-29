"""
Evaluation Module: Computes clustering quality metrics against ground-truth labels.
Supports ARI, NMI, pairwise precision/recall/F1, and cluster purity.
"""

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger("photo_segregator.evaluation")


class Evaluator:
    """
    Computes clustering quality metrics when ground-truth labels are available.

    Metrics:
    - ARI (Adjusted Rand Index): agreement adjusted for chance
    - NMI (Normalized Mutual Information): information-theoretic measure
    - Pairwise Precision / Recall / F1: the gold standard for face clustering
    - Cluster Purity: fraction belonging to dominant identity per cluster
    """

    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Output directory for saving evaluation reports.
        """
        self.output_dir = Path(output_dir) / "_metadata"

    def evaluate(
        self,
        predicted_labels: np.ndarray,
        ground_truth_labels: np.ndarray,
        face_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.

        Args:
            predicted_labels: Predicted cluster labels (int array).
            ground_truth_labels: Ground-truth labels (int array, same length).
            face_ids: Optional face IDs for detailed reporting.

        Returns:
            Dict with all metrics.
        """
        assert len(predicted_labels) == len(ground_truth_labels), (
            f"Label arrays must have same length: "
            f"{len(predicted_labels)} vs {len(ground_truth_labels)}"
        )

        # Filter out noise (-1) from predictions for fair comparison
        valid_mask = predicted_labels >= 0
        pred_valid = predicted_labels[valid_mask]
        gt_valid = ground_truth_labels[valid_mask]
        n_valid = len(pred_valid)
        n_total = len(predicted_labels)
        n_noise = n_total - n_valid

        results = {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_noise": n_noise,
            "noise_ratio": n_noise / n_total if n_total > 0 else 0,
        }

        if n_valid < 2:
            logger.warning("Too few valid predictions for meaningful evaluation")
            results["error"] = "Too few valid predictions"
            return results

        # ARI
        results["ari"] = float(adjusted_rand_score(gt_valid, pred_valid))

        # NMI
        results["nmi"] = float(normalized_mutual_info_score(
            gt_valid, pred_valid, average_method="arithmetic"
        ))

        # Pairwise metrics
        pairwise = self._compute_pairwise_metrics(pred_valid, gt_valid)
        results.update(pairwise)

        # Cluster purity
        purity = self._compute_cluster_purity(pred_valid, gt_valid)
        results.update(purity)

        # Print report
        self._print_report(results)

        # Save report
        self._save_report(results)

        return results

    def _compute_pairwise_metrics(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute pairwise precision, recall, and F1.

        A pair is:
        - TP: same-predicted AND same-ground-truth
        - FP: same-predicted BUT different-ground-truth (false merge)
        - FN: different-predicted BUT same-ground-truth (missed connection)
        - TN: different-predicted AND different-ground-truth
        """
        n = len(predicted)
        tp = fp = fn = tn = 0

        # For efficiency with small datasets, use vectorized approach
        if n <= 1000:
            for i in range(n):
                for j in range(i + 1, n):
                    same_pred = predicted[i] == predicted[j]
                    same_gt = ground_truth[i] == ground_truth[j]

                    if same_pred and same_gt:
                        tp += 1
                    elif same_pred and not same_gt:
                        fp += 1
                    elif not same_pred and same_gt:
                        fn += 1
                    else:
                        tn += 1
        else:
            # Sampling approach for large datasets
            rng = np.random.default_rng(42)
            n_pairs = min(500000, n * (n - 1) // 2)
            indices = rng.choice(n, size=(n_pairs, 2), replace=True)
            indices = indices[indices[:, 0] < indices[:, 1]]  # Remove duplicates

            for i, j in indices:
                same_pred = predicted[i] == predicted[j]
                same_gt = ground_truth[i] == ground_truth[j]

                if same_pred and same_gt:
                    tp += 1
                elif same_pred and not same_gt:
                    fp += 1
                elif not same_pred and same_gt:
                    fn += 1
                else:
                    tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "pairwise_precision": float(precision),
            "pairwise_recall": float(recall),
            "pairwise_f1": float(f1),
            "pairwise_tp": tp,
            "pairwise_fp": fp,
            "pairwise_fn": fn,
            "pairwise_tn": tn,
        }

    def _compute_cluster_purity(
        self, predicted: np.ndarray, ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute cluster purity: for each predicted cluster, what fraction
        of its members belong to the majority ground-truth class.
        """
        unique_pred = np.unique(predicted)
        purities = {}
        weighted_purity = 0.0
        total = len(predicted)

        for cluster in unique_pred:
            mask = predicted == cluster
            cluster_gt = ground_truth[mask]
            cluster_size = len(cluster_gt)

            # Majority class count
            values, counts = np.unique(cluster_gt, return_counts=True)
            majority_count = np.max(counts)
            majority_label = values[np.argmax(counts)]

            purity = majority_count / cluster_size
            purities[int(cluster)] = {
                "purity": float(purity),
                "size": cluster_size,
                "majority_label": int(majority_label),
                "majority_count": int(majority_count),
            }
            weighted_purity += purity * cluster_size

        avg_purity = weighted_purity / total if total > 0 else 0.0

        return {
            "weighted_avg_purity": float(avg_purity),
            "per_cluster_purity": purities,
        }

    def _print_report(self, results: Dict[str, Any]):
        """Print a formatted evaluation report to console and log."""
        report_lines = [
            "",
            "=" * 60,
            "  CLUSTERING EVALUATION REPORT",
            "=" * 60,
            f"  Total faces:        {results['n_total']}",
            f"  Valid (clustered):   {results['n_valid']}",
            f"  Noise (unclustered): {results['n_noise']} ({results['noise_ratio']:.1%})",
            "",
            f"  ARI:                {results.get('ari', 'N/A'):.4f}" if isinstance(results.get('ari'), float) else "",
            f"  NMI:                {results.get('nmi', 'N/A'):.4f}" if isinstance(results.get('nmi'), float) else "",
            "",
            f"  Pairwise Precision: {results.get('pairwise_precision', 'N/A'):.4f}" if isinstance(results.get('pairwise_precision'), float) else "",
            f"  Pairwise Recall:    {results.get('pairwise_recall', 'N/A'):.4f}" if isinstance(results.get('pairwise_recall'), float) else "",
            f"  Pairwise F1:        {results.get('pairwise_f1', 'N/A'):.4f}" if isinstance(results.get('pairwise_f1'), float) else "",
            "",
            f"  Weighted Avg Purity: {results.get('weighted_avg_purity', 'N/A'):.4f}" if isinstance(results.get('weighted_avg_purity'), float) else "",
            "=" * 60,
        ]

        report = "\n".join(line for line in report_lines if line is not None)
        print(report)
        logger.info(report)

    def _save_report(self, results: Dict[str, Any]):
        """Save evaluation report as JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "evaluation_report.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {report_path}")

    @staticmethod
    def load_ground_truth(gt_path: str) -> Dict[str, int]:
        """
        Load ground-truth labels from a JSON file.

        Expected format:
        {
            "face_id_or_image_path": cluster_label (int),
            ...
        }

        Args:
            gt_path: Path to ground-truth JSON file.

        Returns:
            Dict mapping identifier → integer label.
        """
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        return {k: int(v) for k, v in gt.items()}
