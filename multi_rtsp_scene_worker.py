"""Multi-RTSP scene worker: one combined multi-camera request, no local Qwen."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from scene_vlm import Frame, SceneEngine, VisionError, VisionConfig, OpenAICompatibleVLM, append_event, flatten_tasks

_CREDENTIALS_RE = re.compile(r"(://)[^/@\s]+@")


def safe_url(url: str) -> str:
    return _CREDENTIALS_RE.sub(r"\1***@", url)


class RTSPStreamWorker(threading.Thread):
    def __init__(
        self,
        stream_id: str,
        stream_name: str,
        url: str,
        *,
        fps_limit: float = 5.0,
        reconnect_seconds: float = 3.0,
    ) -> None:
        super().__init__(name=f"rtsp-{stream_id}", daemon=True)
        self.stream_id = stream_id
        self.stream_name = stream_name
        self.url = url
        self.fps_limit = fps_limit
        self.reconnect_seconds = reconnect_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest: tuple[float, Image.Image] | None = None
        self.error: str | None = None
        self.connected = False
        self.frames = 0

    def run(self) -> None:
        source: str | int = int(self.url) if self.url.isdigit() else self.url
        while not self._stop.is_set():
            cap = self._open(source)
            if cap is None or not cap.isOpened():
                self.error = f"cannot open {safe_url(self.url)}"
                self._stop.wait(self.reconnect_seconds)
                continue
            self.connected = True
            self.error = None
            try:
                last_store = 0.0
                while not self._stop.is_set():
                    ok, image = cap.read()
                    if not ok:
                        self.error = "stream ended or frame read failed"
                        break
                    self.frames += 1
                    now = time.monotonic()
                    if now - last_store >= 1.0 / self.fps_limit:
                        last_store = now
                        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        pil.thumbnail((768, 768))
                        with self._lock:
                            self.latest = time.perf_counter(), pil
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                cap.release()
                self.connected = False
            if not self._stop.is_set():
                self._stop.wait(self.reconnect_seconds)
        self.connected = False

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> tuple[float, Image.Image] | None:
        with self._lock:
            return self.latest

    def _open(self, source: str | int) -> cv2.VideoCapture | None:
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000]
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, params)
        except TypeError:
            cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap


def test_connection(urls: list[str]) -> bool:
    ok_all = True
    for url in urls:
        source: str | int = int(url) if url.isdigit() else url
        cap = RTSPStreamWorker("test", "test", url)._open(source)
        try:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok and frame is not None:
                height, width = frame.shape[:2]
                print(f"✓ {safe_url(url)} ({width}x{height})")
            else:
                print(f"✗ {safe_url(url)} no frame")
                ok_all = False
        finally:
            cap.release()
    return ok_all


async def run(args: argparse.Namespace) -> None:
    urls = parse_streams(args)
    if not urls:
        raise ValueError("provide --rtsp-url or --rtsp-urls")

    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    event_file = folder / "events.jsonl"
    event_file.touch(exist_ok=True)

    workers = [
        RTSPStreamWorker(stream_id, name, url, fps_limit=args.fps_limit, reconnect_seconds=args.reconnect_seconds)
        for stream_id, name, url in urls
    ]
    for worker in workers:
        worker.start()

    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)

    window_seconds = float(args.window)
    samples_per_window = max(1, int(args.window * args.fps_limit))
    stride_samples = max(1, int(args.stride * args.fps_limit))
    max_total = _env_int("VLM_MAX_FRAMES_PER_REQUEST", 8)
    frames_per_stream = _env_int("VLM_FRAMES_PER_STREAM", max(1, samples_per_window))
    min_confidence = _env_float("VLM_MIN_CONFIDENCE", 0.6)
    votes_required = _env_int("VLM_CONFIRMATION_VOTES", 2)
    settle_seconds = _env_float("VLM_SETTLE_SECONDS", 1.0)
    max_image_side = _env_int("VLM_MAX_IMAGE_SIDE", 768)
    jpeg_quality = _env_int("VLM_JPEG_QUALITY", 80)

    if args.transport_test:
        vlm = None
    else:
        vlm = OpenAICompatibleVLM(VisionConfig.from_env())
    engine = SceneEngine(
        flatten_tasks(_read_task_context(args.task_context)),
        vlm,
        votes_required=votes_required,
        min_confidence=min_confidence,
        settle_seconds=settle_seconds,
        max_image_side=max_image_side,
        jpeg_quality=jpeg_quality,
    )

    samples: dict[str, deque[Frame]] = {}
    slot = 0
    next_window = samples_per_window
    number = 0
    started = time.perf_counter()
    last_received = started
    last_stamps: dict[str, float] = {}

    try:
        while not stopped.is_set():
            now = time.perf_counter()
            if args.seconds and now - started >= args.seconds:
                break
            updated = False
            for worker in workers:
                latest = worker.snapshot()
                if latest and latest[0] != last_stamps.get(worker.stream_id):
                    stamp, image = latest
                    last_stamps[worker.stream_id] = stamp
                    last_received = now
                    frame = Frame(worker.stream_id, worker.stream_name, image.copy(), now - started)
                    samples.setdefault(worker.stream_id, deque(maxlen=max(samples_per_window, frames_per_stream))).append(frame)
                    updated = True

            if updated:
                slot += 1
                if slot >= next_window:
                    frames = combined_frames(samples, frames_per_stream, max_total)
                    if frames:
                        stream_ids = sorted(samples)
                        if args.transport_test:
                            append_event(event_file, "audit", {
                                "status": "TRANSPORT TEST",
                                "model_processed": False,
                                "inference_attempted": False,
                                "sample_count": len(frames),
                                "window_number": number,
                                "streams": stream_ids,
                                "rtspUrls": [safe_url(url) for _, _, url in urls],
                            })
                        else:
                            try:
                                event = await engine.analyze(frames, window_number=number, stream_ids=stream_ids)
                                append_event(event_file, event["kind"], event["data"])
                                if event["kind"] == "task_completed":
                                    print(event["data"]["message"], flush=True)
                            except VisionError as exc:
                                append_event(event_file, "audit", {
                                    "status": "REVIEW",
                                    "model_processed": True,
                                    "inference_attempted": True,
                                    "reason": str(exc),
                                    "sample_count": len(frames),
                                    "streams": stream_ids,
                                    "window_number": number,
                                })
                        number += 1
                        next_window = slot + stride_samples

            if now - last_received > 15:
                raise RuntimeError("No RTSP frames for 15 seconds")
            await asyncio.sleep(1 / args.fps_limit)
    finally:
        for worker in workers:
            worker.stop()
        if vlm is not None:
            await vlm.close()


def combined_frames(samples: dict[str, deque[Frame]], frames_per_stream: int, max_total: int) -> list[Frame]:
    frames: list[Frame] = []
    for stream_id in sorted(samples):
        frames.extend(list(samples[stream_id])[-max(1, frames_per_stream):])
    frames.sort(key=lambda frame: frame.captured_at)
    if len(frames) <= max_total:
        return frames
    return sorted(frames, key=lambda frame: frame.captured_at, reverse=True)[:max_total][::-1]


def parse_streams(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    streams: list[tuple[str, str, str]] = []
    if args.stream:
        for item in args.stream:
            if "=" not in item:
                raise ValueError("--stream must be id=url")
            stream_id, url = item.split("=", 1)
            streams.append((stream_id.strip(), stream_id.strip(), url.strip()))
    if args.rtsp_urls_json:
        payload = json.loads(args.rtsp_urls_json)
        if not isinstance(payload, list):
            raise ValueError("--rtsp-urls-json must be a list")
        for index, item in enumerate(payload):
            if isinstance(item, str):
                streams.append((f"cam{index + 1}", f"cam{index + 1}", item))
            elif isinstance(item, dict) and item.get("url"):
                stream_id = str(item.get("id") or item.get("name") or f"cam{index + 1}")
                streams.append((stream_id, str(item.get("name") or stream_id), str(item["url"])))
    for value in args.rtsp_urls or []:
        streams.append((f"cam{len(streams) + 1}", f"cam{len(streams) + 1}", value))
    if args.rtsp_url:
        streams.append(("cam1" if not streams else f"cam{len(streams) + 1}", f"cam{len(streams) + 1}", args.rtsp_url))
    return streams


def _read_task_context(path: str | None) -> dict[str, Any]:
    if not path:
        return {"topics": []}
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("task context must be a JSON object")
    return payload


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-RTSP scene worker")
    parser.add_argument("--rtsp-url")
    parser.add_argument("--rtsp-urls", nargs="+")
    parser.add_argument("--rtsp-urls-json")
    parser.add_argument("--stream", action="append", help="id=url; repeatable")
    parser.add_argument("--window", type=int, choices=[1, 2, 3, 5], default=3)
    parser.add_argument("--stride", type=int, choices=[1, 2, 3, 5], default=3)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--fps-limit", type=float, default=5.0)
    parser.add_argument("--reconnect-seconds", type=float, default=3.0)
    parser.add_argument("--transport-test", action="store_true")
    parser.add_argument("--task-context")
    parser.add_argument("--output-dir", default="outputs/multi-rtsp")
    parser.add_argument("--job-id", default="multi-rtsp")
    parser.add_argument("--test-connection", action="store_true")
    args = parser.parse_args()
    if args.stride > args.window or args.seconds < 0:
        parser.error("stride must not exceed window; seconds must be nonnegative")

    if args.test_connection:
        streams = parse_streams(args)
        ok = test_connection([url for _, _, url in streams])
        raise SystemExit(0 if ok else 1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
