"""
Photo Segregator — Flask Web Backend

Wraps the existing Python pipeline with a REST API for the web frontend.
Does NOT modify any existing pipeline code — imports and calls existing modules.

Usage:
    python app.py
    # Then open http://localhost:5000 in your browser
"""

import json
import logging
import os
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, Response, send_file

# ── App setup ──────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "photos"
DEFAULT_OUTPUT = BASE_DIR / "output"
CONFIG_PATH = BASE_DIR / "config.yaml"

# ── Pipeline state (shared across threads) ─────────────────────
pipeline_state = {
    "status": "idle",        # idle | running | complete | error
    "progress": 0,           # 0–100
    "current_step": "",
    "step_number": 0,
    "total_steps": 13,
    "message": "",
    "error": None,
    "start_time": None,
    "end_time": None,
    "run_id": None,
}
state_lock = threading.Lock()

# SSE subscribers
sse_queues: list[queue.Queue] = []
sse_lock = threading.Lock()


def broadcast_progress(data: dict):
    """Send a progress event to all SSE subscribers."""
    with sse_lock:
        dead = []
        for q in sse_queues:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_queues.remove(q)


def update_state(**kwargs):
    """Thread-safe state update + broadcast."""
    with state_lock:
        pipeline_state.update(kwargs)
        broadcast_progress({**pipeline_state})


# ── Custom logging handler that captures pipeline progress ────
class ProgressHandler(logging.Handler):
    """Intercepts pipeline log messages to extract step progress."""

    STEP_MAP = {
        "[1/13]": (1, "Initializing components"),
        "[2/13]": (2, "Discovering images"),
        "[3/13]": (3, "Processing images (detect → align → quality → embed)"),
        "[4/13]": (4, "Processing images"),
        "[5/13]": (5, "Processing images"),
        "[6/13]": (6, "Saving embedding cache"),
        "[7/13]": (7, "Semi-supervised classification"),
        "[8/13]": (8, "UMAP dimensionality reduction"),
        "[9/13]": (9, "Computing adaptive thresholds"),
        "[10/13]": (10, "Clustering with HDBSCAN"),
        "[11/13]": (11, "Refining clusters"),
        "[12/13]": (12, "Generating confidence heatmap"),
        "[13/13]": (13, "Writing output folders"),
    }

    def emit(self, record):
        msg = record.getMessage()
        for marker, (step, desc) in self.STEP_MAP.items():
            if marker in msg:
                progress = int((step / 13) * 100)
                update_state(
                    step_number=step,
                    current_step=desc,
                    progress=progress,
                    message=msg.strip(),
                )
                return

        # Forward general messages
        if pipeline_state["status"] == "running":
            update_state(message=msg.strip())


# ── API Routes ─────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/status")
def api_status():
    """Return current pipeline state and output statistics."""
    with state_lock:
        state = {**pipeline_state}

    # Attach output stats if available
    stats = _get_output_stats()
    state["stats"] = stats
    return jsonify(state)


@app.route("/api/run", methods=["POST"])
def api_run():
    """Start the pipeline in a background thread."""
    if pipeline_state["status"] == "running":
        return jsonify({"error": "Pipeline is already running"}), 409

    data = request.get_json(silent=True) or {}
    input_dir = data.get("input_dir", str(DEFAULT_INPUT))
    output_dir = data.get("output_dir", str(DEFAULT_OUTPUT))
    incremental = data.get("incremental", False)

    run_id = str(uuid.uuid4())[:8]
    update_state(
        status="running",
        progress=0,
        current_step="Starting pipeline...",
        step_number=0,
        message="Initializing...",
        error=None,
        start_time=time.time(),
        end_time=None,
        run_id=run_id,
    )

    def _run():
        try:
            from src.utils import load_config, setup_logging
            from main import run_pipeline

            config = load_config(str(CONFIG_PATH))

            # Attach progress handler
            logger = logging.getLogger("photo_segregator")
            handler = ProgressHandler()
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

            # Also set up file logging
            setup_logging(config, output_dir)

            run_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                incremental=incremental,
                logger=logger,
            )

            update_state(
                status="complete",
                progress=100,
                current_step="Pipeline complete",
                message="All done!",
                end_time=time.time(),
            )
            logger.removeHandler(handler)

        except Exception as e:
            update_state(
                status="error",
                message=str(e),
                error=str(e),
                end_time=time.time(),
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return jsonify({"run_id": run_id, "status": "started"})


@app.route("/api/progress")
def api_progress():
    """Server-Sent Events stream for real-time pipeline progress."""
    q = queue.Queue(maxsize=100)
    with sse_lock:
        sse_queues.append(q)

    def stream():
        try:
            # Send initial state
            with state_lock:
                yield f"data: {json.dumps(pipeline_state)}\n\n"
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            with sse_lock:
                if q in sse_queues:
                    sse_queues.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/clusters")
def api_clusters():
    """Return cluster report with photo listings."""
    report_path = DEFAULT_OUTPUT / "_metadata" / "cluster_report.json"
    if not report_path.exists():
        return jsonify({"clusters": {}, "n_clusters": 0})

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Augment each cluster with available thumbnail paths
    for cluster_name, info in report.get("clusters", {}).items():
        cluster_dir = DEFAULT_OUTPUT / cluster_name
        crops_dir = cluster_dir / "crops"

        # Get photo filenames
        photos = []
        if cluster_dir.exists():
            for p in sorted(cluster_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    photos.append(p.name)
        info["photo_files"] = photos

        # Get crop filenames
        crops = []
        if crops_dir.exists():
            for p in sorted(crops_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    crops.append(p.name)
        info["crop_files"] = crops

    return jsonify(report)


@app.route("/api/cluster/<cluster_id>/photos")
def api_cluster_photos(cluster_id):
    """List photos in a specific person cluster."""
    cluster_dir = DEFAULT_OUTPUT / cluster_id
    if not cluster_dir.exists():
        return jsonify({"error": "Cluster not found"}), 404

    photos = []
    for p in sorted(cluster_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            photos.append({
                "name": p.name,
                "size": p.stat().st_size,
                "url": f"/api/photos/{cluster_id}/{p.name}",
            })

    crops = []
    crops_dir = cluster_dir / "crops"
    if crops_dir.exists():
        for p in sorted(crops_dir.iterdir()):
            if p.is_file():
                crops.append({
                    "name": p.name,
                    "url": f"/api/photos/{cluster_id}/crops/{p.name}",
                })

    return jsonify({"cluster_id": cluster_id, "photos": photos, "crops": crops})


@app.route("/api/photos/<path:filepath>")
def api_serve_photo(filepath):
    """Serve a photo from the output directory."""
    full_path = DEFAULT_OUTPUT / filepath
    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "File not found"}), 404

    return send_file(full_path)


@app.route("/api/input-photos/<path:filepath>")
def api_serve_input_photo(filepath):
    """Serve a photo from the input directory."""
    full_path = DEFAULT_INPUT / filepath
    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "File not found"}), 404

    return send_file(full_path)


@app.route("/api/review")
def api_review():
    """Return the review queue."""
    review_path = DEFAULT_OUTPUT / "_review_queue" / "review_items.json"
    if not review_path.exists():
        return jsonify({"items": [], "total": 0})

    with open(review_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Add crop URLs
    for item in items:
        fid = item.get("face_id", "")
        crop_file = DEFAULT_OUTPUT / "_review_queue" / f"{fid}.jpg"
        if crop_file.exists():
            item["crop_url"] = f"/api/photos/_review_queue/{fid}.jpg"
        else:
            item["crop_url"] = None

    return jsonify({"items": items, "total": len(items)})


@app.route("/api/review/<face_id>", methods=["POST"])
def api_review_action(face_id):
    """Submit a review decision for a face."""
    data = request.get_json(silent=True) or {}
    action = data.get("action")  # accept, move, new, discard
    target_cluster = data.get("target_cluster")

    if action not in ("accept", "move", "new", "discard"):
        return jsonify({"error": "Invalid action"}), 400

    # Load existing corrections
    corrections_path = DEFAULT_OUTPUT / "_metadata" / "corrections.json"
    corrections = {}
    if corrections_path.exists():
        with open(corrections_path, "r", encoding="utf-8") as f:
            corrections = json.load(f)

    # Apply correction
    if action == "accept":
        # Load current label from review items
        review_path = DEFAULT_OUTPUT / "_review_queue" / "review_items.json"
        if review_path.exists():
            with open(review_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                if item["face_id"] == face_id:
                    corrections[face_id] = item.get("current_label", -1)
                    break
    elif action == "move":
        if target_cluster is None:
            return jsonify({"error": "target_cluster required for move"}), 400
        corrections[face_id] = int(target_cluster)
    elif action == "new":
        # Find max label
        existing = set(corrections.values())
        report_path = DEFAULT_OUTPUT / "_metadata" / "cluster_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            for name in report.get("clusters", {}):
                try:
                    existing.add(int(name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
        new_label = max(existing) + 1 if existing else 0
        corrections[face_id] = new_label
    elif action == "discard":
        corrections[face_id] = -2

    # Save
    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corrections_path, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=2)

    return jsonify({"face_id": face_id, "action": action, "label": corrections.get(face_id)})


@app.route("/api/heatmap")
def api_heatmap():
    """Serve the cluster heatmap image."""
    heatmap_path = DEFAULT_OUTPUT / "_metadata" / "cluster_heatmap.png"
    if not heatmap_path.exists():
        return jsonify({"error": "Heatmap not generated yet"}), 404
    return send_file(heatmap_path, mimetype="image/png")


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return current config as JSON."""
    from src.utils import load_config
    config = load_config(str(CONFIG_PATH))
    return jsonify(config)


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """Save updated config to config.yaml."""
    import yaml
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No config data"}), 400

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return jsonify({"status": "saved"})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload photos to the input directory."""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    DEFAULT_INPUT.mkdir(parents=True, exist_ok=True)
    uploaded = []
    for f in request.files.getlist("files"):
        if f.filename:
            safe_name = f.filename
            dest = DEFAULT_INPUT / safe_name
            # Avoid overwriting
            if dest.exists():
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = DEFAULT_INPUT / f"{stem}_{counter}{suffix}"
                    counter += 1
            f.save(str(dest))
            uploaded.append(str(dest.name))

    return jsonify({"uploaded": uploaded, "count": len(uploaded)})


@app.route("/api/metadata")
def api_metadata():
    """Return face metadata and threshold report."""
    result = {}

    face_meta_path = DEFAULT_OUTPUT / "_metadata" / "face_metadata.json"
    if face_meta_path.exists():
        with open(face_meta_path, "r", encoding="utf-8") as f:
            result["faces"] = json.load(f)

    threshold_path = DEFAULT_OUTPUT / "_metadata" / "threshold_report.json"
    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            result["thresholds"] = json.load(f)

    return jsonify(result)


@app.route("/api/logs")
def api_logs():
    """Return recent pipeline log lines."""
    log_path = DEFAULT_OUTPUT / "_logs" / "pipeline.log"
    if not log_path.exists():
        return jsonify({"lines": [], "total": 0})

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Return last 200 lines
    recent = lines[-200:]
    return jsonify({"lines": recent, "total": len(lines)})


@app.route("/api/input-photos")
def api_input_photos():
    """List photos in the input directory."""
    if not DEFAULT_INPUT.exists():
        return jsonify({"photos": [], "total": 0})

    photos = []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    for p in sorted(DEFAULT_INPUT.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            photos.append({
                "name": p.name,
                "path": str(p.relative_to(DEFAULT_INPUT)),
                "size": p.stat().st_size,
                "url": f"/api/input-photos/{p.relative_to(DEFAULT_INPUT).as_posix()}",
            })

    return jsonify({"photos": photos, "total": len(photos)})


# ── Helpers ────────────────────────────────────────────────────

def _get_output_stats():
    """Read output directory for quick stats."""
    stats = {
        "n_clusters": 0,
        "n_faces": 0,
        "n_photos": 0,
        "n_review": 0,
        "has_heatmap": False,
        "has_output": False,
    }

    if not DEFAULT_OUTPUT.exists():
        return stats

    stats["has_output"] = True

    # Count person folders
    person_dirs = [d for d in DEFAULT_OUTPUT.iterdir()
                   if d.is_dir() and d.name.startswith("Person_")]
    stats["n_clusters"] = len(person_dirs)

    # Count total photos across clusters
    total_photos = 0
    for pd in person_dirs:
        total_photos += sum(1 for f in pd.iterdir()
                           if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    stats["n_photos"] = total_photos

    # Check cluster report for face count
    report_path = DEFAULT_OUTPUT / "_metadata" / "cluster_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        stats["n_faces"] = report.get("n_faces", 0)
        stats["n_review"] = report.get("n_review", 0)

    # Check heatmap
    stats["has_heatmap"] = (DEFAULT_OUTPUT / "_metadata" / "cluster_heatmap.png").exists()

    # Count input photos
    if DEFAULT_INPUT.exists():
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        stats["n_input_photos"] = sum(
            1 for p in DEFAULT_INPUT.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
    else:
        stats["n_input_photos"] = 0

    return stats


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  +==========================================+")
    print("  |   Photo Segregator -- Web UI             |")
    print("  |   http://localhost:5000                  |")
    print("  +==========================================+\n")
    app.run(debug=True, port=5000, threaded=True)
