"""HTTP manager for the Qwen scene worker.

Same shape as the poseworker manager: Nest starts and stops a child process
over HTTP instead of spawning Python itself. LiveKit mode receives video tracks
and publishes scene-worker.result data; RTSP remains available independently.
"""

from __future__ import annotations

import json
import hmac
import hashlib
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit


BASE_DIR = Path(__file__).resolve().parent
SCRIPT = BASE_DIR / "HyperReality_Qwen2B_Live_RTSP_Developer.py"
RTSP_SCRIPT = BASE_DIR / "multi_rtsp_scene_worker.py"
LIVEKIT_SCRIPT = BASE_DIR / "livekit_scene_worker.py"
OUTPUT_ROOT = BASE_DIR / "outputs" / "HyperReality_Live_RTSP_Qwen2B"
HOST = os.getenv("SCENE_WORKER_HOST", "0.0.0.0")
PORT = int(os.getenv("SCENE_WORKER_PORT", "8090"))
API_TOKEN = os.getenv("SCENE_WORKER_API_TOKEN", "").strip()
MAX_PROCESSES = int(os.getenv("SCENE_WORKER_MAX_PROCESSES", "1"))
STOP_TIMEOUT = float(os.getenv("SCENE_WORKER_STOP_TIMEOUT_SECONDS", "10"))
LOG_LINES = int(os.getenv("SCENE_WORKER_LOG_LINES", "200"))
WINDOWS = {1, 2, 3, 5}
CANCELLATIONS_PATH = BASE_DIR / "outputs" / ".manager-cancellations.json"

LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
if CANCELLATIONS_PATH.is_file():
    CANCELLED_KEYS: dict[str, str] = json.loads(CANCELLATIONS_PATH.read_text(encoding="utf-8"))
else:
    CANCELLED_KEYS = {}


class CancelledKeyError(RuntimeError):
    pass


def remember_cancellation(key: str) -> None:
    """Called under LOCK; retained on disk so delayed starts cannot undo a stop."""
    if key in CANCELLED_KEYS:
        return
    CANCELLED_KEYS[key] = utc_now()
    CANCELLATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CANCELLATIONS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(CANCELLED_KEYS), encoding="utf-8")
    temporary.replace(CANCELLATIONS_PATH)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.username:
        return url
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"***@{host}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def device_info() -> dict:
    info = {"cuda": False, "mps": False, "device": "cpu"}
    try:
        import torch
    except ImportError as exc:
        info["torch"] = f"unavailable: {exc}"
        return info
    info["torch"] = getattr(torch, "__version__", "unknown")
    info["cuda"] = bool(torch.cuda.is_available())
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    info["mps"] = bool(mps and mps.is_available())
    if info["cuda"]:
        info["device"] = "cuda"
    elif info["mps"]:
        info["device"] = "mps"
    return info


def read_latest_events(folder: str | None, limit: int = 20) -> list[dict]:
    if not folder:
        return []
    path = Path(folder) / "events.jsonl"
    if not path.is_file():
        return []
    lines = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if text:
                lines.append(text)
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"kind": "raw", "data": line})
    return events


def public_job(job: dict) -> dict:
    folder = job.get("outputDir")
    if folder and folder != job.get("outputDir"):
        job["outputDir"] = folder
    events = read_latest_events(folder)
    actions = [item.get("data") for item in events if item.get("kind") == "action"]
    completed_tasks = [item.get("data") for item in events if item.get("kind") == "task_completed"]
    return {
        "id": job["id"],
        "idempotencyKey": job.get("idempotencyKey"),
        "status": job["status"],
        "pid": job.get("pid"),
        "mode": job["mode"],
        "rtspUrl": job.get("rtspUrl"),
        "rtspUrls": job.get("rtspUrls"),
        "roomName": job.get("roomName"),
        "participantIdentity": job.get("participantIdentity"),
        "taskContext": job.get("taskContext"),
        "transportTest": job.get("transportTest", False),
        "window": job.get("window"),
        "stride": job.get("stride"),
        "seconds": job.get("seconds"),
        "startedAt": job["startedAt"],
        "stoppedAt": job.get("stoppedAt"),
        "exitCode": job.get("exitCode"),
        "outputDir": folder,
        "latestAction": actions[-1] if actions else None,
        "latestTaskCompleted": completed_tasks[-1] if completed_tasks else None,
        "recentEvents": events,
        "logTail": list(job["log"]),
        "error": job.get("error") or "",
    }


def active_count() -> int:
    return sum(1 for job in JOBS.values() if job["status"] in {"starting", "running", "stopping"})


def append_log(job: dict, line: str) -> None:
    job["log"].append(line.rstrip("\n"))


def watch_process(job: dict, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        with LOCK:
            append_log(job, line)
    code = proc.wait()
    with LOCK:
        job["exitCode"] = code
        job["stoppedAt"] = utc_now()
        if job["status"] in {"stopping", "stopped"}:
            job["status"] = "stopped"
        elif code == 0:
            job["status"] = "exited"
        else:
            job["status"] = "failed"
            job["error"] = job["error"] or f"worker exited with code {code}"


def build_command(body: dict) -> tuple[list[str], dict]:
    mode = str(body.get("mode") or "rtsp").strip().lower()
    if mode == "self-test":
        return [sys.executable, "-u", str(SCRIPT), "--self-test"], {"mode": mode}

    if mode == "livekit":
        url = body.get("url")
        token = body.get("token")
        if not isinstance(url, str) or urlsplit(url).scheme not in {"ws", "wss"}:
            raise ValueError("url must be a LiveKit ws:// or wss:// URL")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("a server-issued LiveKit token is required")
        window = body.get("window", 3)
        stride = body.get("stride", window)
        seconds = body.get("seconds", 0)
        if type(window) is not int or type(stride) is not int or window not in WINDOWS or stride not in WINDOWS:
            raise ValueError("window and stride must be integers 1, 2, 3, or 5")
        if stride > window or type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("stride must not exceed window; seconds must be finite and nonnegative")
        participant = body.get("participantIdentity", "")
        if not isinstance(participant, str) or type(body.get("transportTest", False)) is not bool:
            raise ValueError("participantIdentity must be a string; transportTest must be a boolean")
        context = body.get("taskContext")
        if context is not None:
            validate_task_context(context)
        command = [
            sys.executable, "-u", str(LIVEKIT_SCRIPT), "--url", url,
            "--window", str(window), "--stride", str(stride), "--seconds", str(seconds),
            "--participant-identity", participant,
        ]
        if body.get("transportTest", False):
            command.append("--transport-test")
        return command, {
            "mode": mode, "roomName": body.get("roomName"),
            "participantIdentity": participant, "window": window, "stride": stride,
            "seconds": seconds, "transportTest": body.get("transportTest", False),
        }

    if mode == "test-connection":
        urls = rtsp_urls(body)
        command = [sys.executable, "-u", str(RTSP_SCRIPT), "--test-connection"]
        command.extend(["--rtsp-urls-json", json.dumps([url for _, _, url in urls])])
        return command, {
            "mode": mode,
            "rtspUrl": urls[0][2] and redact_url(urls[0][2]),
            "rtspUrls": [redact_url(url) for _, _, url in urls],
        }

    if mode != "rtsp":
        raise ValueError("mode must be livekit, rtsp, test-connection, or self-test")

    urls = rtsp_urls(body)
    window = int(body.get("window", 3))
    stride = int(body.get("stride", window))
    seconds = float(body.get("seconds", 0))
    if window not in WINDOWS or stride not in WINDOWS:
        raise ValueError("window and stride must be 1, 2, 3, or 5")
    if stride > window:
        raise ValueError("stride cannot exceed window")
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError("seconds must be finite and >= 0")

    command = [
        sys.executable, "-u", str(RTSP_SCRIPT),
        "--window", str(window), "--stride", str(stride), "--seconds", str(seconds),
        "--rtsp-urls-json", json.dumps([url for _, _, url in urls]),
    ]
    if body.get("transportTest"):
        command.append("--transport-test")
    context = body.get("taskContext")
    if context is not None:
        validate_task_context(context)
    return command, {
        "mode": mode,
        "rtspUrl": redact_url(urls[0][2]),
        "rtspUrls": [redact_url(url) for _, _, url in urls],
        "window": window,
        "stride": stride,
        "seconds": seconds,
        "transportTest": bool(body.get("transportTest", False)),
    }


def validate_task_context(context: dict) -> None:
    if not isinstance(context, dict) or not isinstance(context.get("topics"), list):
        raise ValueError("taskContext must contain a topics array")
    for topic in context["topics"]:
        if not isinstance(topic, dict) or not isinstance(topic.get("title"), str) or not isinstance(topic.get("tasks"), list):
            raise ValueError("each topic must have a title and tasks array")
        for task in topic["tasks"]:
            if not isinstance(task, dict) or not isinstance(task.get("title"), str) or not isinstance(task.get("instruction"), str):
                raise ValueError("each task must have a title and instruction")
    if not any(topic["tasks"] for topic in context["topics"]):
        raise ValueError("taskContext must contain at least one task")


def rtsp_urls(body: dict) -> list[tuple[str, str, str]]:
    streams: list[tuple[str, str, str]] = []
    single = str(body.get("rtspUrl") or "").strip()
    if single:
        streams.append(("cam1", "cam1", single))
    multiple = body.get("rtspUrls")
    if multiple is not None:
        if not isinstance(multiple, list) or not multiple:
            raise ValueError("rtspUrls must be a nonempty list")
        for index, item in enumerate(multiple):
            if isinstance(item, str):
                url = item.strip()
                stream_id = f"cam{index + 1}"
                name = stream_id
            elif isinstance(item, dict):
                url = str(item.get("url") or "").strip()
                stream_id = str(item.get("id") or item.get("name") or f"cam{index + 1}")
                name = str(item.get("name") or stream_id)
            else:
                raise ValueError("rtspUrls entries must be strings or objects with url")
            streams.append((stream_id, name, url))
    if not streams:
        raise ValueError("provide rtspUrl or rtspUrls")
    for _, _, url in streams:
        if not (url.startswith("rtsp://") or url.startswith("rtsps://")):
            raise ValueError("RTSP URLs must start with rtsp:// or rtsps://")
    return streams


def start_job(body: dict) -> dict:
    if not SCRIPT.is_file():
        raise RuntimeError(f"worker script missing: {SCRIPT}")
    command, meta = build_command(body)
    context = body.get("taskContext")
    context_bytes = json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8") if context is not None else b""
    context_hash = hashlib.sha256(context_bytes).hexdigest()
    key = body.get("idempotencyKey")
    if key is not None and (not isinstance(key, str) or not key or len(key) > 512):
        raise ValueError("idempotencyKey must be a nonempty string no longer than 512 characters")
    with LOCK:
        if key:
            if key in CANCELLED_KEYS:
                raise CancelledKeyError("This worker start was cancelled; use a new idempotencyKey for a new attempt")
            for existing in JOBS.values():
                if existing.get("idempotencyKey") == key and existing["status"] in {"starting", "running", "stopping"}:
                    if existing["contextHash"] != context_hash:
                        raise CancelledKeyError("Active job has different task context")
                    return public_job(existing)
        if active_count() >= MAX_PROCESSES:
            raise RuntimeError(f"scene worker is already running (max {MAX_PROCESSES})")
        job = {
            "id": uuid.uuid4().hex,
            "idempotencyKey": key,
            "contextHash": context_hash,
            "status": "starting",
            "pid": None,
            "startedAt": utc_now(),
            "stoppedAt": None,
            "exitCode": None,
            "outputDir": None,
            "log": deque(maxlen=LOG_LINES),
            "error": "",
            "process": None,
            **meta,
        }
        JOBS[job["id"]] = job
        environment = os.environ.copy()
        if meta["mode"] in {"rtsp", "livekit"}:
            folder = BASE_DIR / "outputs" / job["id"]
            job["outputDir"] = str(folder)
            command.extend(["--output-dir", str(folder)])
        if meta["mode"] == "livekit":
            command.extend(["--job-id", job["id"]])
            environment["SCENE_WORKER_LIVEKIT_TOKEN"] = body["token"]
        try:
            if meta["mode"] in {"livekit", "rtsp"} and context is not None:
                folder.mkdir(parents=True, exist_ok=False)
                context_path = folder / "task-context.json"
                descriptor = os.open(context_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(context, handle, ensure_ascii=False)
                command.extend(["--task-context", str(context_path)])
            proc = subprocess.Popen(
                command, cwd=str(BASE_DIR), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True,
            )
        except OSError as exc:
            job.update(status="failed", error=str(exc), stoppedAt=utc_now())
            raise RuntimeError(f"Cannot start worker: {exc}") from exc
        job["process"] = proc
        job["pid"] = proc.pid
        job["status"] = "running"
    threading.Thread(target=watch_process, args=(job, proc), daemon=True).start()
    return public_job(job)


def stop_job(job_id: str) -> dict:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("idempotencyKey"):
            remember_cancellation(job["idempotencyKey"])
        proc = job.get("process")
        if job["status"] not in {"starting", "running"} or proc is None or proc.poll() is not None:
            return public_job(job)
        job["status"] = "stopping"
        pid = proc.pid
    try:
        os.killpg(pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)
    with LOCK:
        job["status"] = "stopped"
        job["exitCode"] = proc.returncode
        job["stoppedAt"] = job["stoppedAt"] or utc_now()
        return public_job(job)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s %s\n" % (utc_now(), fmt % args))

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        if not API_TOKEN or self.path.split("?", 1)[0] == "/healthz":
            return True
        header = self.headers.get("Authorization", "")
        return hmac.compare_digest(header, f"Bearer {API_TOKEN}")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length > 4 * 1024 * 1024 or length < 0:
            raise ValueError("body must be no larger than 4 MiB")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(401, {"error": "unauthorized", "status": 401})
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/healthz":
            self._json(200, {
                "status": "ok",
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "script": SCRIPT.is_file(),
                **device_info(),
            })
            return
        if path == "/jobs":
            key = parse_qs(urlsplit(self.path).query).get("key", [None])[0]
            with LOCK:
                jobs = [public_job(job) for job in JOBS.values() if key is None or job.get("idempotencyKey") == key]
            self._json(200, {"jobs": jobs})
            return
        if path.startswith("/jobs/"):
            job_id = path.split("/", 2)[2]
            with LOCK:
                job = JOBS.get(job_id)
                payload = public_job(job) if job else None
            if payload is None:
                self._json(404, {"error": "job not found", "status": 404})
                return
            self._json(200, payload)
            return
        self._json(404, {"error": "not found", "status": 404})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(401, {"error": "unauthorized", "status": 401})
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/jobs":
            self._json(404, {"error": "not found", "status": 404})
            return
        try:
            body = self._read_json()
            job = start_job(body)
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"invalid JSON: {exc.msg}", "status": 400})
            return
        except (ValueError, TypeError, OverflowError) as exc:
            self._json(400, {"error": str(exc), "status": 400})
            return
        except CancelledKeyError as exc:
            self._json(409, {"error": str(exc), "status": 409})
            return
        except RuntimeError as exc:
            self.log_message("worker start failed: %s", exc)
            self._json(503, {"error": str(exc), "status": 503})
            return
        self._json(201, job)

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._authorized():
            self._json(401, {"error": "unauthorized", "status": 401})
            return
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/jobs":
            key = parse_qs(urlsplit(self.path).query).get("key", [""])[0]
            if not key or len(key) > 512:
                self._json(400, {"error": "a valid key query parameter is required", "status": 400})
                return
            with LOCK:
                try:
                    remember_cancellation(key)
                except OSError as exc:
                    self.log_message("cannot persist worker cancellation: %s", exc)
                    self._json(503, {"error": "cannot persist worker cancellation", "status": 503})
                    return
                ids = [job["id"] for job in JOBS.values() if job.get("idempotencyKey") == key]
            jobs = [stop_job(job_id) for job_id in ids]
            self._json(200, {"cancelled": True, "idempotencyKey": key, "jobs": jobs})
            return
        if not path.startswith("/jobs/"):
            self._json(404, {"error": "not found", "status": 404})
            return
        job_id = path.split("/", 2)[2]
        try:
            payload = stop_job(job_id)
        except KeyError:
            self._json(404, {"error": "job not found", "status": 404})
            return
        except OSError as exc:
            self.log_message("cannot stop worker: %s", exc)
            self._json(503, {"error": "cannot stop worker", "status": 503})
            return
        self._json(200, payload)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"scene-worker manager listening on {HOST}:{PORT}", flush=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping scene-worker manager and active jobs", flush=True)
    finally:
        with LOCK:
            active_ids = [job["id"] for job in JOBS.values() if job["status"] in {"starting", "running"}]
        for job_id in active_ids:
            stop_job(job_id)
        server.server_close()


if __name__ == "__main__":
    main()
