"""HyperReality live webcam DEVELOPER BENCHMARK with UNCHANGED Qwen3-VL-2B logic.

Run:
    python HyperReality_Qwen2B_Live_Webcam_Developer.py --self-test
    python HyperReality_Qwen2B_Live_Webcam_Developer.py --camera 0
    python HyperReality_Qwen2B_Live_Webcam_Developer.py --camera 0 --seconds 60

Camera captures continuously in the main thread. A separate worker runs the original
core.load_qwen / core.infer_once / core.parse_one. Every result, rejection, and skipped
window is journaled. On exit, a review workbook is generated with the original writer.

Important: predictions are UNVERIFIED, latency is not guaranteed, and this is not
continuous action tracking across windows. No YOLO, cloud API, or master approval.
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import hashlib
import importlib.util
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from importlib.metadata import version as library_version
except ImportError:
    library_version = None

CORE_FILE = Path(__file__).with_name("HyperReality_Final_Action_Review.py")
EXPECTED_SHA256 = "bff7d33ad27f7f6ee30ed0c19aa7da8fbd85171ceca0a914a9e0927a8727f4af"
WINDOW_SECONDS = 5.0
SAMPLE_FPS = 2
SAMPLES_PER_WINDOW = int(WINDOW_SECONDS * SAMPLE_FPS)
OUTPUT_BASE = Path("outputs") / "HyperReality_Live_Qwen2B"


def original_core():
    if not CORE_FILE.is_file():
        raise FileNotFoundError(f"Original Qwen 2B script missing: {CORE_FILE}")
    digest = hashlib.sha256(CORE_FILE.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "Original 2B script differs from the verified baseline. "
            "Use the HyperReality_Final_Action_Review.py included in the package; "
            "the live runner deliberately does not change it."
        )
    spec = importlib.util.spec_from_file_location("hyperreality_qwen2b_original", CORE_FILE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.MODEL == "Qwen/Qwen3-VL-2B-Instruct"
    assert mod.WINDOW == WINDOW_SECONDS and mod.SAMPLE_FPS == SAMPLE_FPS
    return mod


def clean_text(value: str, limit: int = 125) -> str:
    return " ".join(str(value).split())[:limit]


@dataclass
class WindowJob:
    number: int
    start: float
    end: float
    frames: list[Any]
    sample_paths: list[str]
    enqueued_at: float
    capture_times: list[float] | None = None
    queue_depth_on_submit: int = 0


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]):
    """Create an ordinary Excel-readable CSV; no extra spreadsheet dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = {}
            for key in fieldnames:
                value = row.get(key, "")
                if value is None:
                    value = ""
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    value = "'" + value  # Avoid formula injection when CSV is opened in Excel.
                safe[key] = value
            writer.writerow(safe)


def metric_seconds(value):
    return round(value, 4) if isinstance(value, (float, int)) else None


def current_gpu_snapshot():
    """Read only optional CUDA counters; do not change model or inference."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),
            "gpu_session_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
        }
    except Exception:
        return {}


def process_rss_mb():
    try:
        import psutil  # Optional; do not require pip install.
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024**2), 1)
    except Exception:
        return None


def append_jsonl(path: Path, item: dict):
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        out.flush()


class LiveJournal:
    def __init__(self, folder: Path, core, start_wall: str):
        self.folder = folder
        self.core = core
        self.start_wall = start_wall
        self.file = folder / "events.jsonl"
        self.lock = threading.Lock()
        self.latest = "No prediction yet"
        self.total_candidates = 0
        self.total_skipped = 0
        self.last_inference = 0.0
        self.start_perf: float | None = None

    def add(self, audit: dict, action: dict | None = None):
        with self.lock:
            append_jsonl(self.file, {"kind": "audit", "data": audit})
            if action is not None:
                append_jsonl(self.file, {"kind": "action", "data": action})
                self.total_candidates += 1
                self.latest = action["action"]
            elif audit["status"] == "NO ACTION":
                self.latest = "NO CLEAR ACTION"
            elif audit["status"] == "REVIEW":
                self.latest = "REVIEW: invalid model output"
            elif audit["status"].startswith("SKIPPED"):
                self.total_skipped += 1
            self.last_inference = float(audit.get("seconds") or 0)

    def elapsed(self, at_perf: float | None = None):
        return metric_seconds((at_perf if at_perf is not None else time.perf_counter()) - self.start_perf) if self.start_perf is not None else None

    def snapshot(self):
        with self.lock:
            return self.latest, self.total_candidates, self.total_skipped, self.last_inference


def record_action(core, job: WindowJob, result: dict, raw: str, seconds: float) -> dict:
    """Same timestamp arithmetic and action wording as core.candidate_record.

    Uses actual captured sample paths for thumbnails, since a live camera may skip
    a sample and old core.image_at assumes a complete offline video cache.
    """
    a = min(job.end, job.start + result["first"] / SAMPLE_FPS)
    b = min(job.end, job.start + result["last"] / SAMPLE_FPS)
    if b <= a:
        b = min(job.end, a + 0.5)
    form = core.object_phrase(result["part"])
    first = min(SAMPLES_PER_WINDOW - 1, max(0, result["first"]))
    last = min(SAMPLES_PER_WINDOW - 1, max(first, result["last"] - 1))
    mid = min(SAMPLES_PER_WINDOW - 1, (first + last) // 2)
    return {
        "id": f"e{job.number+1:04d}", "start": a, "end": b,
        "timestamp": core.interval(a, b), "action_type": result["verb"],
        "action": f"{core.past_tense(result['verb']).capitalize()} {form}",
        "part": result["part"], "tool": result["tool"],
        "tool_setting": "", "fixture": result["fixture"],
        "instructions": f"{result['verb'].capitalize()} {form}",
        "context": result["evidence"], "status": "UNVERIFIED — HUMAN REVIEW",
        "raw": raw, "processing_seconds": seconds,
        "before": job.sample_paths[max(0, first-1)],
        "during": job.sample_paths[mid],
        "after": job.sample_paths[min(SAMPLES_PER_WINDOW-1, last+1)],
        "video_duration": job.end,
    }


def run_one(core, job: WindowJob, infer: Callable, journal: LiveJournal):
    """Instrument original infer_once() without changing Qwen inputs/generation."""
    worker_started = time.perf_counter()
    gpu_before = current_gpu_snapshot()
    memory_before = process_rss_mb()
    begun = time.perf_counter()  # Start clock IMMEDIATELY before unchanged model call.
    tokens = None
    try:
        output = infer(list(enumerate(job.frames)))
        if isinstance(output, tuple):
            raw = output[0]
            tokens = output[1] if len(output) > 1 else None
        else:
            raw = output
    except Exception as exc:
        raw = f"[INFERENCE_ERROR] {type(exc).__name__}: {exc}"
    completed = time.perf_counter()  # Stop clock IMMEDIATELY after model returns.
    elapsed = round(completed - begun, 3)  # Wall time includes model preprocess/generation/decode.
    gpu_after = current_gpu_snapshot()
    memory_after = process_rss_mb()
    parsed, status, note = core.parse_one(raw, len(job.frames))
    action = record_action(core, job, parsed, raw, elapsed) if parsed is not None else None
    sample_times = job.capture_times or []
    gaps = [b-a for a,b in zip(sample_times, sample_times[1:])]
    end_to_end = journal.elapsed(completed)
    audit = {
        "timeframe": core.interval(job.start, job.end),
        "status": status, "reason": note,
        "summary": action["action"] if action else "",
        "raw": raw, "seconds": elapsed,
        "window_number": job.number,
        "window_start_seconds": job.start,
        "window_end_seconds": job.end,
        "sample_count": len(job.frames),
        "samples_expected": SAMPLES_PER_WINDOW,
        "sample_timestamps_seconds": [metric_seconds(t) for t in sample_times],
        "sample_paths": job.sample_paths,
        "first_sample_seconds": metric_seconds(sample_times[0]) if sample_times else None,
        "last_sample_seconds": metric_seconds(sample_times[-1]) if sample_times else None,
        "avg_sampling_gap_seconds": metric_seconds(statistics.mean(gaps)) if gaps else None,
        "max_sampling_gap_seconds": metric_seconds(max(gaps)) if gaps else None,
        "model_input_edge_pixels": core.MODEL_EDGE,
        "preview_edge_pixels": core.PREVIEW_EDGE,
        "queue_depth_when_queued": job.queue_depth_on_submit,
        "enqueued_session_seconds": journal.elapsed(job.enqueued_at),
        "worker_start_session_seconds": journal.elapsed(worker_started),
        "pre_inference_setup_seconds": metric_seconds(max(0.0, begun-worker_started)),
        "inference_start_session_seconds": journal.elapsed(begun),
        "inference_end_session_seconds": end_to_end,
        "queue_delay_seconds": metric_seconds(max(0.0, worker_started - job.enqueued_at)),
        "inference_seconds": elapsed,
        "response_delay_after_window_seconds": metric_seconds(max(0.0, end_to_end-job.end)) if end_to_end is not None else None,
        "output_tokens": tokens,
        "output_tokens_per_second": round(tokens/elapsed, 2) if isinstance(tokens,int) and elapsed > 0 else None,
        "process_rss_mb_before": memory_before,
        "process_rss_mb_after": memory_after,
        "gpu_allocated_mb_before": gpu_before.get("gpu_allocated_mb"),
        "gpu_allocated_mb_after": gpu_after.get("gpu_allocated_mb"),
        "gpu_reserved_mb_after": gpu_after.get("gpu_reserved_mb"),
        "gpu_session_peak_allocated_mb": gpu_after.get("gpu_session_peak_allocated_mb"),
        "model_processed": not str(raw).startswith("[INFERENCE_ERROR]"),
        "inference_attempted": True,
    }
    journal.add(audit, action)
    lag = audit["response_delay_after_window_seconds"]
    lag_txt = f"{lag:.2f}s" if lag is not None else "n/a"
    print(f"{audit['timeframe']} | {status} | frames {len(job.frames)}/{SAMPLES_PER_WINDOW} "
          f"| Qwen {elapsed:.2f}s | queue {audit['queue_delay_seconds']:.2f}s "
          f"| result lag {lag_txt} | "
          f"{action['action'] if action else clean_text(raw, 85)}", flush=True)


def worker_loop(core, jobs: queue.Queue, journal: LiveJournal, infer):
    while True:
        job = jobs.get()
        try:
            if job is None:
                return
            run_one(core, job, infer, journal)
        finally:
            jobs.task_done()


def skip_window(core, journal, number, reason, status="SKIPPED", sample_count=0,
                sample_times=None, sample_paths=None):
    start = number * WINDOW_SECONDS
    end = start + WINDOW_SECONDS
    journal.add({
        "timeframe": core.interval(start, end),
        "status": status, "reason": reason,
        "summary": "", "raw": "", "seconds": 0,
        "window_number": number, "window_start_seconds": start, "window_end_seconds": end,
        "sample_count": sample_count, "samples_expected": SAMPLES_PER_WINDOW,
        "sample_timestamps_seconds": sample_times or [],
        "sample_paths": sample_paths or [],
        "model_processed": False, "inference_attempted": False, "inference_seconds": None,
        "output_tokens": None, "queue_delay_seconds": None,
        "response_delay_after_window_seconds": None,
    })
    print(f"{core.interval(start, end)} | {status} | frames {sample_count}/{SAMPLES_PER_WINDOW} "
          f"| NOT ANALYZED | {reason}", flush=True)


def submit_freshest(core, jobs: queue.Queue, journal: LiveJournal, job: WindowJob):
    try:
        jobs.put_nowait(job)
        return
    except queue.Full:
        pass
    # Never block the camera on Qwen inference; do not pretend a dropped window
    # was analyzed. Keep the freshest complete window queued.
    try:
        old = jobs.get_nowait()
    except queue.Empty:
        old = None
    else:
        jobs.task_done()
        if old is not None:
            skip_window(core, journal, old.number,
                        "Model was busy; replaced queued window with newer observations",
                        "SKIPPED BACKLOG", len(old.frames), old.capture_times, old.sample_paths)
    try:
        jobs.put_nowait(job)
    except queue.Full:
        skip_window(core, journal, job.number, "Model queue still full", "SKIPPED BACKLOG",
                    len(job.frames), job.capture_times, job.sample_paths)


def read_events(path: Path):
    audit, actions = [], []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("kind") == "audit":
                audit.append(entry["data"])
            elif entry.get("kind") == "action":
                actions.append(entry["data"])
    audit.sort(key=lambda x: (x.get("window_number", 0), x["timeframe"]))
    actions.sort(key=lambda x: (x["start"], x["end"]))
    return audit, actions


METRIC_COLUMNS = [
    "window_number", "timeframe", "status", "reason", "model_processed", "inference_attempted", "summary",
    "window_start_seconds", "window_end_seconds", "sample_count", "samples_expected",
    "first_sample_seconds", "last_sample_seconds", "sample_timestamps_seconds",
    "avg_sampling_gap_seconds", "max_sampling_gap_seconds", "sample_paths",
    "model_input_edge_pixels", "preview_edge_pixels", "queue_depth_when_queued",
    "enqueued_session_seconds", "worker_start_session_seconds", "pre_inference_setup_seconds",
    "inference_start_session_seconds", "inference_end_session_seconds",
    "queue_delay_seconds", "inference_seconds",
    "response_delay_after_window_seconds", "output_tokens", "output_tokens_per_second",
    "process_rss_mb_before", "process_rss_mb_after", "gpu_allocated_mb_before",
    "gpu_allocated_mb_after", "gpu_reserved_mb_after", "gpu_session_peak_allocated_mb",
    "raw"
]


def metric_stats(values):
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not clean:
        return {"count": 0, "min": None, "mean": None, "median": None, "p95": None, "max": None}
    # Nearest-rank p95 instead of NumPy dependency.
    p95 = clean[max(0, math.ceil(.95 * len(clean)) - 1)]
    return {"count": len(clean), "min": round(clean[0], 3),
            "mean": round(statistics.mean(clean), 3),
            "median": round(statistics.median(clean), 3),
            "p95": round(p95, 3), "max": round(clean[-1], 3)}


def export_developer_diagnostics(session_dir: Path, info: dict, audit: list[dict]):
    """Full per-window audit, plus compact benchmark summary. No model execution."""
    metrics_file = session_dir / "Developer_Window_Metrics.csv"
    write_csv_rows(metrics_file, METRIC_COLUMNS, audit)
    processed = [a for a in audit if a.get("model_processed", False)]
    attempted = [a for a in audit if a.get("inference_attempted", False)]
    skipped = [a for a in audit if str(a.get("status", "")).startswith(("SKIPPED", "INCOMPLETE"))]
    stats = {
        "description": "Developer benchmark of Qwen3-VL-2B live inference; no simulated timings in real sessions",
        "model": info.get("model"), "baseline_sha256": info.get("original_script_sha256"),
        "session_start": info.get("started_at"), "session_end": info.get("ended_at"),
        "session_duration_seconds": info.get("elapsed_seconds"),
        "source_fps_reported": info.get("source_fps"),
        "camera_read_frames_total": info.get("camera_read_frames_total"),
        "camera_read_fps_measured": info.get("camera_read_fps_measured"),
        "camera_samples_saved_total": info.get("camera_samples_saved_total"),
        "camera_samples_fps_measured": info.get("camera_samples_fps_measured"),
        "camera_samples_missed_slots": info.get("camera_samples_missed_slots"),
        "camera_width": info.get("width"), "camera_height": info.get("height"),
        "requested_sample_fps": info.get("sample_fps"),
        "window_seconds": info.get("window_seconds"),
        "model_load_seconds": info.get("model_load_seconds"),
        "camera_read_seconds": info.get("camera_read_seconds_stats"),
        "camera_recording_file": info.get("camera_recording_file"),
        "machine": info.get("machine", {}),
        "total_windows_audited": len(audit), "analyzed_windows": len(processed),
        "inference_attempts": len(attempted),
        "candidate_windows": sum(a.get("status") == "CANDIDATE" for a in audit),
        "no_action_windows": sum(a.get("status") == "NO ACTION" for a in audit),
        "review_windows": sum(a.get("status") == "REVIEW" for a in audit),
        "skipped_or_incomplete_windows": len(skipped),
        "skipped_backlog_windows": sum(a.get("status") == "SKIPPED BACKLOG" for a in audit),
        "skipped_capture_windows": sum(a.get("status") == "SKIPPED CAPTURE" for a in audit),
        "inference_time_seconds": metric_stats([a.get("inference_seconds") for a in attempted]),
        "queue_wait_seconds": metric_stats([a.get("queue_delay_seconds") for a in attempted]),
        "pre_inference_setup_seconds": metric_stats([a.get("pre_inference_setup_seconds") for a in attempted]),
        "response_delay_after_window_seconds": metric_stats([a.get("response_delay_after_window_seconds") for a in attempted]),
        "generated_output_tokens": sum(a.get("output_tokens") or 0 for a in processed),
        "error_windows": sum("[INFERENCE_ERROR]" in str(a.get("raw", "")) for a in audit),
        "note": "Action timestamps and labels are model proposals. Response-delay is measured from "
                "the 5-second window boundary to inference completion. "
                "The worker may not meet the 5-second live throughput requirement. "
                "Model input is sampled at 2 FPS; missing samples are not fabricated."
    }
    (session_dir / "Developer_Session_Summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    t = stats["inference_time_seconds"]
    lag = stats["response_delay_after_window_seconds"]
    lines = [
        "HYPERREALITY / LIVE QWEN2B / DEVELOPER BENCHMARK",
        "================================================",
        f"Model: {stats['model']}",
        f"Session duration: {stats['session_duration_seconds']} s",
        f"Window: {stats['window_seconds']} s | samples requested: 10 (2 FPS)",
        f"Camera: {stats['camera_width']}x{stats['camera_height']}",
        f"Camera FPS (reported/measured): {stats['source_fps_reported']} / {stats['camera_read_fps_measured']}",
        f"Model startup / loading time: {stats['model_load_seconds']} s",
        f"Window totals: {stats['total_windows_audited']} audited; {stats['analyzed_windows']} analyzed; "
            f"{stats['inference_attempts']} attempted inference; "
            f"{stats['skipped_backlog_windows']} skipped due to backlog; "
            f"{stats['skipped_capture_windows']} skipped due to missing samples",
        f"Inference: mean={t['mean']} s; median={t['median']} s; p95={t['p95']} s; max={t['max']} s",
        f"Queue wait: mean={stats['queue_wait_seconds']['mean']} s; p95={stats['queue_wait_seconds']['p95']} s",
        f"Result delay after window ends: mean={lag['mean']} s; p95={lag['p95']} s; max={lag['max']} s",
        "Interpretation: If median inference exceeds 5 s, one worker cannot sustain every incoming 5 s window.",
        "Per-window details: Developer_Window_Metrics.csv",
        "Per-sample details: Developer_Frame_Samples.csv",
        "Full model outputs + review: events.jsonl and Live_Qwen2B_Review.xlsx",
        "Skipped windows were NOT analyzed and have blank inference-time metrics.",
        "Outputs are unverified model suggestions, not ground truth."
    ]
    (session_dir / "Developer_Read_Me_First.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("DEVELOPER METRICS:", metrics_file)
    print("DEVELOPER SUMMARY:", session_dir / "Developer_Read_Me_First.txt")
    return stats


def export_review(core, session_dir: Path):
    events = session_dir / "events.jsonl"
    meta_path = session_dir / "session.json"
    if not events.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Session needs both events.jsonl and session.json")
    info = json.loads(meta_path.read_text(encoding="utf-8"))
    audit, actions = read_events(events)
    elapsed = float(info.get("elapsed_seconds", 0))
    if audit:
        elapsed = max(elapsed, (max(x.get("window_number", 0) for x in audit) + 1) * WINDOW_SECONDS)
    # The original workbook writer expects the source video name and duration.
    virtual_video = Path(info.get("source_name", "Live_Webcam.mp4"))
    virtual_info = {"duration": elapsed, "fps": info.get("source_fps", 0),
                    "width": info.get("width", 0), "height": info.get("height", 0)}
    out = session_dir / "Live_Qwen2B_Review.xlsx"
    core.create_review_book(out, virtual_video, virtual_info, actions, audit)
    export_developer_diagnostics(session_dir, info, audit)
    print(f"\nEXCEL: {out}")
    print(f"Actions proposed: {len(actions)} | Total audited windows: {len(audit)}")
    print("All results are unverified; no master workbook was modified.")
    return out


def write_session_meta(folder: Path, meta: dict):
    path = folder / "session.json"
    temp = folder / "session.json.tmp"
    temp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(temp, path)


def show_overlay(frame, cv2, elapsed: float, message: str, candidates: int, skipped: int, backlog: int):
    height, width = frame.shape[:2]
    display = frame.copy()
    cv2.rectangle(display, (0, 0), (width, min(height, 105)), (12, 27, 43), -1)
    cv2.putText(display, f"LIVE {elapsed:06.1f}s  |  Q to finish", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 245, 255), 2)
    cv2.putText(display, f"Actions: {candidates}  Skipped: {skipped}  Queued: {backlog}",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (198, 221, 242), 1)
    cv2.putText(display, clean_text(message, max(25, int(width/9))), (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (95, 255, 184), 1)
    return display


def machine_metadata():
    data = {"platform": platform.platform(), "python": sys.version.split()[0],
            "pid": os.getpid(), "cpu_count_logical": os.cpu_count()}
    if library_version:
        for package in ("torch", "transformers", "opencv-python", "Pillow", "qwen-vl-utils"):
            try:
                data[package] = library_version(package)
            except Exception:
                pass
    try:
        import torch
        data["cuda_available"] = torch.cuda.is_available()
        data["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            data["gpu_name"] = gpu.name
            data["gpu_total_vram_mb"] = round(gpu.total_memory / (1024**2), 1)
    except Exception as exc:
        data["gpu_probe_error"] = str(exc)
    return data


def live_camera(core, camera_index: int, seconds: float, show_preview: bool,
                output: Path, record_video: bool = False):
    import cv2
    from PIL import Image

    # Windows: DirectShow first if available, otherwise platform default.
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}; try --camera 1 or check Windows camera privacy settings")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        print("Loading offline Qwen3-VL-2B once; capture starts after it loads...", flush=True)
        load_begin = time.perf_counter()
        model, processor = core.load_qwen()
        load_seconds = metric_seconds(time.perf_counter() - load_begin)
        print(f"Qwen loaded in {load_seconds:.2f}s. Starting LIVE webcam capture. Press Q or ESC to stop.", flush=True)
        folder = output
        folder.mkdir(parents=True, exist_ok=True)
        frames_dir = folder / "sampled_images"
        frames_dir.mkdir(exist_ok=True)
        recording = None
        recording_path = folder / "Developer_Camera_Recording.mp4"
        if record_video:
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            safe_fps = source_fps if 5 <= source_fps <= 120 else 30.0
            recording = cv2.VideoWriter(str(recording_path), codec, safe_fps, (width, height))
            if not recording.isOpened():
                raise RuntimeError("Could not create optional video recording; run without --record-video or check OpenCV codecs")
            print(f"Developer raw camera recording: {recording_path}")
        session_meta = {
            "source_name": f"Live_Webcam_Camera_{camera_index}.mp4",
            "camera_index": camera_index,
            "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_fps": source_fps, "width": width, "height": height,
            "sample_fps": SAMPLE_FPS, "window_seconds": WINDOW_SECONDS,
            "original_script_sha256": EXPECTED_SHA256,
            "model": core.MODEL, "elapsed_seconds": 0.0,
            "model_load_seconds": load_seconds,
            "camera_recording_file": str(recording_path.resolve()) if recording else None,
            "machine": machine_metadata(),
            "note": "Source is a live webcam, not a saved file. Output proposals are unverified."
        }
        write_session_meta(folder, session_meta)
        journal = LiveJournal(folder, core, session_meta["started_at"])
        jobs = queue.Queue(maxsize=1)
        def model_infer(frames):
            return core.infer_once(model, processor, frames)  # Same inference; include token count for diagnostics.
        worker = threading.Thread(target=worker_loop, args=(core, jobs, journal, model_infer), daemon=False)
        worker.start()
        started = None
        last_slot = -1
        next_window = 0
        sampled = {}  # Only the unfinished window is retained in RAM.
        sample_paths = {}
        sample_timestamps = {}
        last_elapsed = 0.0
        missing_frames = 0
        read_frames = 0
        saved_samples = 0
        read_times = []  # Camera-grab API call latency (separate from Qwen infer time).
        frame_samples_csv = folder / "Developer_Frame_Samples.csv"
        frame_sample_cols = ["sample_slot", "window_number", "index_within_window",
                             "expected_slot_seconds", "actual_capture_seconds",
                             "sample_timing_offset_ms", "capture_wall_time_utc", "image_path",
                             "camera_width", "camera_height", "saved_width", "saved_height",
                             "model_edge_pixels"]
        frame_file = frame_samples_csv.open("w", newline="", encoding="utf-8-sig")
        frame_writer = csv.DictWriter(frame_file, fieldnames=frame_sample_cols)
        frame_writer.writeheader()
        try:
            while True:
                read_begun = time.perf_counter()
                ok, frame = cap.read()
                sampled_at = time.perf_counter()
                read_times.append(sampled_at - read_begun)
                if not ok:
                    print("Camera stopped returning frames; ending session.")
                    break
                if started is None:
                    started = sampled_at
                    journal.start_perf = started
                read_frames += 1
                if recording:
                    recording.write(frame)
                elapsed = max(0.0, sampled_at - started)
                last_elapsed = elapsed
                slot = int(elapsed * SAMPLE_FPS)
                if slot > last_slot:
                    if last_slot >= 0:
                        missing_frames += max(0, slot - last_slot - 1)
                    last_slot = slot
                    # Save actual observed camera imagery only (never interpolate blanks).
                    h, w = frame.shape[:2]
                    scale = min(1.0, core.PREVIEW_EDGE / max(w, h))
                    cache_im = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))),
                                          interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
                    name = frames_dir / f"f{slot+1:06d}.jpg"
                    if not cv2.imwrite(str(name), cache_im):
                        raise RuntimeError(f"Cannot save actual sampled frame {name}")
                    sample_paths[slot] = str(name.resolve())
                    sample_timestamps[slot] = elapsed
                    saved_samples += 1
                    frame_writer.writerow({
                        "sample_slot": slot, "window_number": slot // SAMPLES_PER_WINDOW,
                        "index_within_window": slot % SAMPLES_PER_WINDOW,
                        "expected_slot_seconds": round(slot / SAMPLE_FPS, 4),
                        "actual_capture_seconds": round(elapsed, 4),
                        "sample_timing_offset_ms": round(1000 * (elapsed - slot / SAMPLE_FPS), 2),
                        "capture_wall_time_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                        "image_path": str(name.resolve()), "camera_width": w,
                        "camera_height": h, "saved_width": cache_im.shape[1],
                        "saved_height": cache_im.shape[0], "model_edge_pixels": core.MODEL_EDGE,
                    })
                    frame_file.flush()  # Survives a crash mid-session.
                    scale_model = min(1.0, core.MODEL_EDGE / max(w, h))
                    small = cv2.resize(frame, (max(1, round(w*scale_model)),
                                               max(1, round(h*scale_model))),
                                       interpolation=cv2.INTER_AREA) if scale_model < 1.0 else frame
                    sampled[slot] = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

                finished_windows = int(elapsed // WINDOW_SECONDS)
                while next_window < finished_windows:
                    win = next_window
                    indices = list(range(win*SAMPLES_PER_WINDOW, (win+1)*SAMPLES_PER_WINDOW))
                    if not all(idx in sampled for idx in indices):
                        absent = sum(idx not in sampled for idx in indices)
                        actual_indices = [i for i in indices if i in sampled]
                        skip_window(core, journal, win, f"Missing {absent}/10 camera samples",
                                    "SKIPPED CAPTURE", len(actual_indices),
                                    [sample_timestamps[i] for i in actual_indices],
                                    [sample_paths[i] for i in actual_indices])
                    else:
                        job = WindowJob(win, win*WINDOW_SECONDS, (win+1)*WINDOW_SECONDS,
                                        [sampled[i] for i in indices],
                                        [sample_paths[i] for i in indices], time.perf_counter(),
                                        [sample_timestamps[i] for i in indices], jobs.qsize())
                        submit_freshest(core, jobs, journal, job)
                    for idx in indices:
                        sampled.pop(idx, None)
                        sample_paths.pop(idx, None)
                        sample_timestamps.pop(idx, None)
                    next_window += 1
                if show_preview:
                    latest, count, skipped, _ = journal.snapshot()
                    img = show_overlay(frame, cv2, elapsed, latest, count, skipped, jobs.qsize())
                    cv2.imshow("HyperReality LIVE - Qwen3-VL-2B (Q to finish)", img)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                        break
                if seconds > 0 and elapsed >= seconds:
                    break
        except KeyboardInterrupt:
            print("Keyboard interrupt — finishing and exporting captured results")
        finally:
            # Capture stops first; no heavy model work is performed in capture loop.
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            frame_file.close()
            if recording:
                recording.release()
            if sampled:
                remaining = sorted(sampled)
                skip_window(core, journal, next_window,
                            "Session ended before full 5-second window completed", "INCOMPLETE WINDOW",
                            len(remaining), [sample_timestamps[i] for i in remaining],
                            [sample_paths[i] for i in remaining])
            if missing_frames:
                print(f"Camera samples missed: {missing_frames} (see SKIPPED CAPTURE audit entries)")
            jobs.put(None)  # This can wait for the final queued inference; camera is closed.
            jobs.join()
            worker.join()
            session_meta["elapsed_seconds"] = round(last_elapsed, 3)
            session_meta["camera_read_frames_total"] = read_frames
            session_meta["camera_read_fps_measured"] = round(read_frames/last_elapsed, 3) if last_elapsed else None
            session_meta["camera_samples_saved_total"] = saved_samples
            session_meta["camera_samples_fps_measured"] = round(saved_samples/last_elapsed, 3) if last_elapsed else None
            session_meta["camera_samples_missed_slots"] = missing_frames
            session_meta["camera_read_seconds_stats"] = metric_stats(read_times)
            session_meta["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            write_session_meta(folder, session_meta)
            export_review(core, folder)
    finally:
        if cap.isOpened():
            cap.release()


def self_test(core):
    """Real workbook creation with synthetic webcam observations; no Qwen or camera."""
    import tempfile
    from PIL import Image
    import zipfile
    with tempfile.TemporaryDirectory(prefix="hyperreality_live_test_") as tmp:
        folder = Path(tmp)
        meta = {"source_name": "Live_Webcam_Camera_0.mp4", "elapsed_seconds": 10,
                "width": 640, "height": 480, "source_fps": 30}
        write_session_meta(folder, meta)
        images = []
        paths = []
        for i in range(20):
            path = folder / f"frame_{i:02}.jpg"
            img = Image.new("RGB", (320, 240), (i*9 % 255, 50, 90))
            img.save(path)
            images.append(img)
            paths.append(str(path))
        journal = LiveJournal(folder, core, "TEST")
        journal.start_perf = time.perf_counter()
        job0 = WindowJob(0, 0, 5, images[:10], paths[:10], time.perf_counter(),
                         [x/2 for x in range(10)], 0)
        job1 = WindowJob(1, 5, 10, images[10:], paths[10:], time.perf_counter(),
                         [5+x/2 for x in range(10)], 0)
        run_one(core, job0,
                lambda f: ("move|wheel|||1|7|The technician moves the visible wheel away from its original position.", 26),
                journal)
        run_one(core, job1, lambda f: "NONE", journal)
        skip_window(core, journal, 2, "Simulated model backlog", "SKIPPED BACKLOG")
        assert core.PROMPT.startswith("You are reviewing 10 visual observations")
        audit, actions = read_events(journal.file)
        assert len(actions) == 1 and actions[0]["timestamp"] == "0:00 - 0:04", actions
        assert [x["status"] for x in audit] == ["CANDIDATE", "NO ACTION", "SKIPPED BACKLOG"]
        review = export_review(core, folder)
        assert review.exists()
        with zipfile.ZipFile(review) as workbook:
            assert workbook.testzip() is None
        rows = core.xlsx_values(review, "Actions")
        assert any(x.get("C") == "Moved the wheel" for x in rows.values())
        audit_sheet = core.xlsx_values(review, "Raw Audit")
        assert any(x.get("B") == "NO ACTION" for x in audit_sheet.values())
        assert any(x.get("B") == "SKIPPED BACKLOG" for x in audit_sheet.values())
        summary = json.loads((folder / "Developer_Session_Summary.json").read_text())
        assert summary["total_windows_audited"] == 3
        assert summary["analyzed_windows"] == 2 and summary["skipped_backlog_windows"] == 1
        assert summary["generated_output_tokens"] == 26
        with (folder / "Developer_Window_Metrics.csv").open(encoding="utf-8-sig", newline="") as data:
            metrics = list(csv.DictReader(data))
        assert len(metrics) == 3 and metrics[0]["sample_count"] == "10"
        assert metrics[0]["output_tokens"] == "26"
        assert metrics[2]["inference_seconds"] == "", "Skipped is not a 0s inference"
        assert float(metrics[0]["inference_seconds"]) >= 0
        print("SELF-TEST PASS: original Qwen prompt/parser, action times, detailed per-window "
              "timings/frames/tokens, candidate/NONE/skips, thumbnails, CSV, summary and XLSX")


def main():
    parser = argparse.ArgumentParser(description="HyperReality Qwen 2B LIVE webcam + detailed developer diagnostics (unchanged model logic)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index, usually 0")
    parser.add_argument("--seconds", type=float, default=0, help="Stop after N seconds; 0 means Q/ESC")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview, use Ctrl+C to stop")
    parser.add_argument("--record-video", action="store_true", help="Optionally save the full webcam video for ground-truth review; adds I/O overhead")
    parser.add_argument("--self-test", action="store_true", help="No camera or Qwen model required")
    parser.add_argument("--rebuild", type=Path, help="Rebuild Excel from an existing session directory, no GPU")
    args = parser.parse_args()
    core = original_core()
    if args.self_test:
        self_test(core)
        return
    if args.rebuild:
        export_review(core, args.rebuild)
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_BASE / f"Webcam_{args.camera}_{stamp}"
    live_camera(core, args.camera, args.seconds, not args.no_preview,
                session_dir, record_video=args.record_video)


if __name__ == "__main__":
    main()
