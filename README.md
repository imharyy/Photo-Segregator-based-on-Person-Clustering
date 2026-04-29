# Photo Segregator

> **Production-ready, offline-first photo grouping by person** using state-of-the-art face detection (RetinaFace), recognition (ArcFace), and density-based clustering (HDBSCAN).

## Features

- **RetinaFace** face detection with 5-point landmark extraction
- **ArcFace** 512-d face embeddings (InsightFace `buffalo_l` model pack)
- **Ensemble embeddings** — averaging original, flipped, and perturbed crops (most impactful accuracy improvement)
- **UMAP** dimensionality reduction before clustering (reduces noise in 512-d space)
- **HDBSCAN** primary clustering with **DBSCAN** fallback
- **Adaptive thresholding** — computes merge/split/review thresholds from dataset statistics
- **Semi-supervised learning** — uses corrected labels from review to improve future runs
- **Confidence heatmap** — 2D UMAP scatter plot for visual debugging
- **Interactive CLI** for reviewing uncertain face-cluster assignments
- **Incremental processing** — only process new images when added to the folder
- **Multi-face support** — same photo appears in multiple person folders
- **Quality filtering** — handles blur, occlusion, masks, side faces, tiny faces
- **Full evaluation suite** — ARI, NMI, pairwise F1, cluster purity
- **GPU acceleration** (NVIDIA CUDA) with automatic CPU fallback
- **100% offline** — no network calls, all data stays local

## Quick Start

### 1. Prerequisites

- Python 3.8+
- NVIDIA GPU recommended (works on CPU too)

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install core packages
pip install numpy opencv-python Pillow PyYAML tqdm scikit-learn scikit-image matplotlib seaborn
pip install onnxruntime umap-learn easydict setuptools wheel Cython pytest

# Install insightface (may need patching on Python 3.13+)
pip install insightface
```

> **Note on Python 3.13+**: If `insightface` fails to build due to a Cython mesh extension, download the source, remove the `ext_modules` from `setup.py`, and install locally. The Cython extension is only for 3D face rendering — not needed for detection or recognition. See `venv/tmp_dl/` for the patched version.

> **Note on HDBSCAN**: This project uses `sklearn.cluster.HDBSCAN` (included in scikit-learn ≥ 1.3) instead of the standalone `hdbscan` package, avoiding C compilation issues.

> **Note on GPU**: If you have NVIDIA CUDA installed, install `onnxruntime-gpu` instead of `onnxruntime` for GPU acceleration. To check providers: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`

> **Note on first run**: InsightFace will automatically download the `buffalo_l` model pack (~300MB) on the first run. This requires an internet connection the first time only.

### 3. Run the Pipeline

```bash
# Basic usage: process a folder of photos
python main.py --input ./photos --output ./output

# With custom config
python main.py --input ./photos --output ./output --config config.yaml

# Incremental mode (only process new images)
python main.py --input ./photos --output ./output --incremental
```

### 4. Review Uncertain Faces

```bash
python main.py --review --output ./output
```

This launches an interactive CLI where you can:
- **[a]ccept** the current cluster assignment
- **[m]ove** a face to a different cluster
- **[n]ew** — create a new cluster for the face
- **[d]iscard** — mark as non-face / garbage
- **[s]kip** — leave for later
- **[q]uit** — save and exit

After review, re-run the pipeline to apply learned corrections.

### 5. Evaluate Clustering Quality

```bash
# Requires a ground-truth JSON file
python main.py --evaluate --output ./output --ground-truth ./labels.json
```

Ground-truth format (`labels.json`):
```json
{
    "image1.jpg": 0,
    "image2.jpg": 0,
    "image3.jpg": 1,
    "image4.jpg": 2
}
```

### 6. Calibrate Parameters

```bash
python main.py --calibrate --input ./photos --ground-truth ./labels.json
```

This runs a grid search over HDBSCAN parameters and reports the best settings.

## Output Structure

```
output/
├── Person_000/
│   ├── photo1.jpg           # Original image (copied)
│   ├── photo5.jpg
│   └── crops/               # Aligned 112×112 face crops
│       ├── abc123_000.jpg
│       └── def456_001.jpg
├── Person_001/
│   └── ...
├── _review_queue/
│   ├── review_items.json    # Faces needing manual review
│   └── face_crops...
├── _metadata/
│   ├── face_metadata.json   # Complete face → cluster mapping
│   ├── cluster_report.json  # Cluster statistics
│   ├── embeddings.npz       # Cached embeddings (for incremental)
│   ├── threshold_report.json # Adaptive threshold details
│   ├── cluster_heatmap.png  # Visual debugging heatmap
│   └── known_identities.npz # Learned identity centroids
└── _logs/
    └── pipeline.log         # Detailed processing log
```

## Configuration

All parameters are in `config.yaml`. Key tunable settings:

| Parameter | Default | Description |
|---|---|---|
| `detection.det_thresh` | 0.5 | Face detection confidence threshold |
| `quality.min_face_size` | 40 | Minimum face size in pixels |
| `quality.blur_threshold` | 50.0 | Laplacian variance threshold for blur |
| `ensemble.enabled` | true | Enable ensemble embedding averaging |
| `umap.n_components` | 50 | UMAP target dimensionality |
| `clustering.hdbscan.min_cluster_size` | 2 | Min faces to form a cluster |
| `adaptive_threshold.enabled` | true | Use dataset-aware thresholds |

## Architecture

```
Image → RetinaFace Detection → 5-Point Alignment → Quality Filter
    → ArcFace Embedding → Ensemble Averaging → Embedding Cache
    → UMAP Reduction → Adaptive Thresholds → HDBSCAN Clustering
    → Cluster Refinement → Confidence Scoring → Folder Writer
                                              → Review Queue
                                              → Confidence Heatmap
```

## Pipeline Modules

| Module | File | Purpose |
|---|---|---|
| Image Loader | `src/image_loader.py` | Discover and load JPG/PNG images |
| Face Detector | `src/face_detector.py` | RetinaFace detection + landmarks |
| Face Aligner | `src/face_aligner.py` | 5-point landmark alignment to 112×112 |
| Quality Filter | `src/quality_filter.py` | Blur, size, pose, occlusion checks |
| Embedding Extractor | `src/embedding_extractor.py` | ArcFace 512-d embeddings |
| Ensemble Embeddings | `src/ensemble_embeddings.py` | Multi-view averaging |
| Embedding Cache | `src/embedding_cache.py` | Persistent disk storage |
| Dimensionality Reduction | `src/dimensionality_reduction.py` | UMAP 512→50 |
| Adaptive Threshold | `src/adaptive_threshold.py` | Dataset-aware thresholds |
| Clustering | `src/clustering.py` | HDBSCAN + DBSCAN fallback |
| Cluster Refinement | `src/cluster_refinement.py` | Merge/split/reassign |
| Semi-Supervised | `src/semi_supervised.py` | Learn from corrections |
| Confidence Heatmap | `src/confidence_heatmap.py` | 2D UMAP visualization |
| Folder Writer | `src/folder_writer.py` | Output directory creation |
| Review CLI | `src/review_cli.py` | Interactive review tool |
| Evaluation | `src/evaluation.py` | ARI, NMI, F1, purity metrics |

## Improving Accuracy

1. **Ensemble embeddings** (enabled by default) — most impactful single improvement
2. **UMAP before clustering** — reduces curse of dimensionality
3. **Adaptive thresholds** — avoids one-size-fits-all parameters
4. **Review → re-cluster loop** — semi-supervised learning improves over time
5. **Calibrate on labeled data** — use `--calibrate` for optimal parameters
6. **Lower `det_thresh`** for more recall (catches occluded/side faces)
7. **Adjust `min_face_size`** based on your photo resolution

## Privacy & Security

- **Fully offline**: No network calls after initial model download
- **Local-first**: All processing happens on your machine
- **Non-reversible**: Face embeddings cannot reconstruct original faces
- **Configurable cleanup**: Delete embeddings after processing via cache
- **No telemetry**: Zero data collection

## License

This project uses the InsightFace library (MIT License). Note that InsightFace pre-trained models are for **non-commercial research purposes** by default. See [insightface.ai](https://insightface.ai) for commercial licensing.
