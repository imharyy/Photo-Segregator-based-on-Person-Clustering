"""
Unit tests for the Photo Segregator pipeline.
Tests individual modules with mock/simple data.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ============================================================
# Test Utilities
# ============================================================

class TestUtils:
    """Tests for src/utils.py"""

    def test_load_default_config(self):
        from src.utils import load_config
        config = load_config("nonexistent_config.yaml")
        assert "detection" in config
        assert "clustering" in config
        assert config["detection"]["model_pack"] == "buffalo_l"

    def test_compute_image_hash(self, tmp_path):
        from src.utils import compute_image_hash
        # Create a test file
        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")
        
        hash1 = compute_image_hash(str(test_file))
        hash2 = compute_image_hash(str(test_file))
        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 64  # SHA-256

    def test_format_face_id(self):
        from src.utils import format_face_id
        face_id = format_face_id("abcdef1234567890abcdef", 5)
        assert face_id == "abcdef1234567890_005"

    def test_ensure_dir(self, tmp_path):
        from src.utils import ensure_dir
        new_dir = tmp_path / "a" / "b" / "c"
        result = ensure_dir(str(new_dir))
        assert result.exists()
        assert result.is_dir()


# ============================================================
# Test Image Loader
# ============================================================

class TestImageLoader:
    """Tests for src/image_loader.py"""

    def test_discover_images(self, tmp_path):
        from src.image_loader import discover_images
        # Create test files
        (tmp_path / "photo1.jpg").write_bytes(b"fake")
        (tmp_path / "photo2.png").write_bytes(b"fake")
        (tmp_path / "document.pdf").write_bytes(b"fake")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "photo3.JPG").write_bytes(b"fake")

        paths = discover_images(str(tmp_path), recursive=True)
        extensions = {p.suffix.lower() for p in paths}
        assert ".pdf" not in extensions
        assert len(paths) >= 2  # At least jpg and png

    def test_discover_empty_dir(self, tmp_path):
        from src.image_loader import discover_images
        paths = discover_images(str(tmp_path))
        assert len(paths) == 0

    def test_discover_nonexistent_dir(self):
        from src.image_loader import discover_images
        with pytest.raises(FileNotFoundError):
            discover_images("/nonexistent/path")


# ============================================================
# Test Quality Filter
# ============================================================

class TestQualityFilter:
    """Tests for src/quality_filter.py"""

    def test_size_check_pass(self):
        from src.quality_filter import QualityFilter
        config = {"quality": {"min_face_size": 40}}
        qf = QualityFilter(config)
        
        bbox = np.array([0, 0, 100, 100])
        landmarks = np.array([[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], dtype=np.float32)
        face_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = qf.evaluate(face_crop, bbox, landmarks, 0.9)
        assert result.scores["size"] > 0.5

    def test_size_check_fail(self):
        from src.quality_filter import QualityFilter
        config = {"quality": {"min_face_size": 40}}
        qf = QualityFilter(config)
        
        bbox = np.array([0, 0, 15, 15])  # Very small
        landmarks = np.array([[5, 5], [10, 5], [7, 8], [5, 12], [10, 12]], dtype=np.float32)
        face_crop = np.random.randint(0, 255, (15, 15, 3), dtype=np.uint8)
        
        result = qf.evaluate(face_crop, bbox, landmarks, 0.9)
        assert result.scores["size"] < 0.5

    def test_confidence_penalty(self):
        from src.quality_filter import QualityFilter, QualityResult
        qf = QualityFilter({})
        
        good_result = QualityResult(passed=True, composite_score=0.9, scores={}, reasons=[])
        assert qf.get_confidence_penalty(good_result) == 1.0
        
        bad_result = QualityResult(passed=False, composite_score=0.3, scores={}, reasons=["blurry"])
        assert qf.get_confidence_penalty(bad_result) < 1.0


# ============================================================
# Test Face Aligner
# ============================================================

class TestFaceAligner:
    """Tests for src/face_aligner.py"""

    def test_align_face(self):
        from src.face_aligner import align_face
        
        # Create a dummy image
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        # Landmarks roughly matching a face
        landmarks = np.array([
            [60, 80],   # left eye
            [140, 80],  # right eye
            [100, 110], # nose
            [70, 140],  # left mouth
            [130, 140], # right mouth
        ], dtype=np.float32)
        
        aligned = align_face(image, landmarks)
        assert aligned is not None
        assert aligned.shape == (112, 112, 3)

    def test_align_insufficient_landmarks(self):
        from src.face_aligner import align_face
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        landmarks = np.array([[60, 80], [140, 80]], dtype=np.float32)  # Only 2 points
        
        aligned = align_face(image, landmarks)
        assert aligned is None


# ============================================================
# Test Embedding Cache
# ============================================================

class TestEmbeddingCache:
    """Tests for src/embedding_cache.py"""

    def test_add_and_retrieve(self, tmp_path):
        from src.embedding_cache import EmbeddingCache
        
        cache = EmbeddingCache(str(tmp_path))
        embedding = np.random.randn(512).astype(np.float32)
        metadata = {"image_path": "test.jpg", "image_hash": "abc123"}
        
        cache.add_face("face_001", embedding, metadata)
        
        assert cache.count == 1
        retrieved = cache.get_embedding("face_001")
        assert retrieved is not None
        assert np.allclose(embedding, retrieved.flatten())

    def test_save_and_load(self, tmp_path):
        from src.embedding_cache import EmbeddingCache
        
        cache = EmbeddingCache(str(tmp_path))
        for i in range(5):
            emb = np.random.randn(512).astype(np.float32)
            cache.add_face(f"face_{i:03d}", emb, {"image_hash": f"hash_{i}"})
        cache.save()
        
        # Load in new instance
        cache2 = EmbeddingCache(str(tmp_path))
        cache2.load()
        assert cache2.count == 5

    def test_incremental(self, tmp_path):
        from src.embedding_cache import EmbeddingCache
        
        cache = EmbeddingCache(str(tmp_path))
        cache.add_face("f1", np.random.randn(512).astype(np.float32), {"image_hash": "h1"})
        cache.save()
        
        # Reload and add more
        cache2 = EmbeddingCache(str(tmp_path))
        cache2.load()
        cache2.add_face("f2", np.random.randn(512).astype(np.float32), {"image_hash": "h2"})
        cache2.save()
        
        assert cache2.count == 2
        assert cache2.is_image_processed("h1") is False  # Hash function checks file, not raw hash


# ============================================================
# Test Dimensionality Reduction
# ============================================================

class TestDimensionalityReducer:
    """Tests for src/dimensionality_reduction.py"""

    def test_disabled(self):
        from src.dimensionality_reduction import DimensionalityReducer
        config = {"umap": {"enabled": False}}
        reducer = DimensionalityReducer(config)
        
        embeddings = np.random.randn(20, 512)
        result = reducer.fit_transform(embeddings)
        assert result.shape == embeddings.shape  # Unchanged

    def test_too_few_samples(self):
        from src.dimensionality_reduction import DimensionalityReducer
        config = {"umap": {"enabled": True}}
        reducer = DimensionalityReducer(config)
        
        embeddings = np.random.randn(2, 512)
        result = reducer.fit_transform(embeddings)
        assert result.shape == embeddings.shape  # Unchanged (too few)


# ============================================================
# Test Adaptive Threshold
# ============================================================

class TestAdaptiveThreshold:
    """Tests for src/adaptive_threshold.py"""

    def test_compute_thresholds(self):
        from src.adaptive_threshold import AdaptiveThreshold
        config = {"adaptive_threshold": {"enabled": True}}
        at = AdaptiveThreshold(config)
        
        # Create embeddings with known structure
        embeddings = np.random.randn(30, 512).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        thresholds = at.compute(embeddings)
        assert "merge_threshold" in thresholds
        assert "split_threshold" in thresholds
        assert "review_threshold" in thresholds
        assert all(v > 0 for v in thresholds.values())

    def test_disabled(self):
        from src.adaptive_threshold import AdaptiveThreshold
        config = {"adaptive_threshold": {"enabled": False}}
        at = AdaptiveThreshold(config)
        
        embeddings = np.random.randn(10, 512).astype(np.float32)
        thresholds = at.compute(embeddings)
        assert "merge_threshold" in thresholds


# ============================================================
# Test Clustering
# ============================================================

class TestClustering:
    """Tests for src/clustering.py"""

    def test_cluster_basic(self):
        from src.clustering import ClusteringEngine
        config = {"clustering": {
            "primary_method": "hdbscan",
            "fallback_method": "dbscan",
            "noise_fallback_ratio": 0.6,
            "hdbscan": {"min_cluster_size": 2, "min_samples": 1, "metric": "euclidean"},
            "dbscan": {"eps": 0.5, "min_samples": 2, "metric": "euclidean"},
        }}
        engine = ClusteringEngine(config)
        
        # Create 3 tight clusters with clear separation
        rng = np.random.default_rng(42)
        offsets = np.zeros(50)
        
        offsets_1 = offsets.copy(); offsets_1[0] = 10
        offsets_2 = offsets.copy(); offsets_2[0] = -10
        offsets_3 = offsets.copy(); offsets_3[1] = 10
        
        cluster1 = rng.standard_normal((10, 50)) * 0.5 + offsets_1
        cluster2 = rng.standard_normal((10, 50)) * 0.5 + offsets_2
        cluster3 = rng.standard_normal((10, 50)) * 0.5 + offsets_3
        embeddings = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)
        
        labels, probs = engine.cluster(embeddings)
        assert len(labels) == 30
        assert len(set(labels) - {-1}) >= 2  # At least 2 clusters

    def test_single_embedding(self):
        from src.clustering import ClusteringEngine
        config = {"clustering": {
            "hdbscan": {"min_cluster_size": 2, "min_samples": 1, "metric": "euclidean"},
            "dbscan": {"eps": 0.5, "min_samples": 2, "metric": "euclidean"},
        }}
        engine = ClusteringEngine(config)
        
        labels, probs = engine.cluster(np.random.randn(1, 50))
        assert labels[0] == 0
        assert probs[0] == 1.0


# ============================================================
# Test Cluster Refinement
# ============================================================

class TestClusterRefinement:
    """Tests for src/cluster_refinement.py"""

    def test_confidence_scoring(self):
        from src.cluster_refinement import ClusterRefiner
        config = {"refinement": {
            "merge_threshold": 0.3,
            "split_threshold": 0.8,
            "reassign_threshold": 0.6,
            "review_threshold": 0.3,
        }}
        refiner = ClusterRefiner(config)
        
        embeddings = np.random.randn(10, 512).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, -1, -1])
        probs = np.array([0.9, 0.8, 0.7, 0.9, 0.6, 0.5, 0.8, 0.7, 0.1, 0.05])
        face_ids = [f"face_{i}" for i in range(10)]
        
        refined_labels, confidences, review = refiner.refine(
            embeddings, labels, probs, face_ids
        )
        
        assert len(refined_labels) == 10
        assert len(confidences) == 10
        assert isinstance(review, list)


# ============================================================
# Test Evaluation
# ============================================================

class TestEvaluation:
    """Tests for src/evaluation.py"""

    def test_perfect_clustering(self, tmp_path):
        from src.evaluation import Evaluator
        evaluator = Evaluator(str(tmp_path))
        
        predicted = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        ground_truth = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        results = evaluator.evaluate(predicted, ground_truth)
        assert results["ari"] == 1.0
        assert results["nmi"] == 1.0
        assert results["pairwise_f1"] == 1.0
        assert results["weighted_avg_purity"] == 1.0

    def test_random_clustering(self, tmp_path):
        from src.evaluation import Evaluator
        evaluator = Evaluator(str(tmp_path))
        
        predicted = np.array([0, 1, 0, 1, 0, 1])
        ground_truth = np.array([0, 0, 0, 1, 1, 1])
        
        results = evaluator.evaluate(predicted, ground_truth)
        # Random clustering should have low metrics
        assert results["pairwise_f1"] < 1.0


# ============================================================
# Run tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
