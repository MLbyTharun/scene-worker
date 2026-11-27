"""HyperReality Qwen3-VL-2B live RTSP developer benchmark.

Keeps HyperReality_Final_Action_Review.py and its Qwen weights, generation, action
parser and XLSX writer untouched. Reuses the measured live developer pipeline.
Only stream capture and per-window sample-count wording vary for 3-second tests.

Examples:
 python HyperReality_Qwen2B_Live_RTSP_Developer.py --self-test
 python HyperReality_Qwen2B_Live_RTSP_Developer.py --rtsp rtsp://127.0.0.1:8554/hyperreality --test-connection
 python HyperReality_Qwen2B_Live_RTSP_Developer.py --rtsp rtsp://127.0.0.1:8554/hyperreality --window 5 --stride 5 --seconds 60
 python HyperReality_Qwen2B_Live_RTSP_Developer.py --rtsp rtsp://127.0.0.1:8554/hyperreality --window 3 --stride 3 --seconds 60
 python HyperReality_Qwen2B_Live_RTSP_Developer.py --rtsp rtsp://127.0.0.1:8554/hyperreality --window 5 --stride 2 --seconds 60

Camera timestamps here are the receiver's monotonic wall-clock arrival times,
NOT source capture PTS. True one-way network latency needs synchronized source
capture timestamps; it cannot be inferred from an RTSP stream alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import HyperReality_Qwen2B_Live_Webcam_Developer as dev

FPS = 2  # Identical to original Qwen inference sampling rate.
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
OUTPUT_BASE = Path('outputs') / 'HyperReality_Live_RTSP_Qwen2B'


def redact_url(url: str) -> str:
    u = urlsplit(url)
    if not u.username:
        return url
    host = u.hostname or ''
    if ':' in host and not host.startswith('['):
        host = f'[{host}]'
    netloc = f'***@{host}'
    if u.port:
        netloc += f':{u.port}'
    return urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))


@dataclass
class StreamSample:
    number: int
    received_perf: float
    pixels: bytes


class FFmpegRTSPReader:
    """Read FFmpeg's 2fps decoded BGR24 image stream without blocking Qwen."""
    def __init__(self, url: str, width: int, height: int, folder: Path):
        self.url, self.width, self.height = url, width, height
        self.frame_size = width * height * 3
        self.samples: queue.Queue[StreamSample] = queue.Queue(maxsize=24)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.stderr_file = (folder / 'FFmpeg_RTSP_Decode_Log.txt').open('w', encoding='utf-8')
        self.frames_decoded = 0
        self.dropped_in_reader_queue = 0
        self.eof = False
        self.started_perf: float | None = None
        self.first_frame_perf: float | None = None

    def start(self):
        if not shutil.which('ffmpeg'):
            raise RuntimeError('FFmpeg not found on PATH; run ffmpeg -version first')
        vf = (f'fps={FPS},scale={self.width}:{self.height}:'
              f'force_original_aspect_ratio=decrease,'
              f'pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2')
        command = [
            'ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'warning',
            '-rtsp_transport', 'tcp', '-i', self.url, '-an', '-sn',
            '-vf', vf, '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'
        ]
        self.started_perf = time.perf_counter()
        # shell=False, so URLs/credentials are not interpolated into a shell command.
        self.proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                     stderr=self.stderr_file, bufsize=0)
        self.thread = threading.Thread(target=self._read_loop, name='ffmpeg-rtsp-reader', daemon=True)
        self.thread.start()

    def _read_loop(self):
        try:
            while not self.stop_event.is_set():
                chunks = bytearray()
                while len(chunks) < self.frame_size and not self.stop_event.is_set():
                    piece = self.proc.stdout.read(self.frame_size - len(chunks))
                    if not piece:
                        self.eof = True
                        return
                    chunks.extend(piece)
                if len(chunks) != self.frame_size:
                    return
                received = time.perf_counter()
                if self.first_frame_perf is None:
                    self.first_frame_perf = received
                s = StreamSample(self.frames_decoded, received, bytes(chunks))
                self.frames_decoded += 1
                try:
                    self.samples.put_nowait(s)
                except queue.Full:
                    try:
                        self.samples.get_nowait()
                        self.samples.task_done()
                        self.dropped_in_reader_queue += 1
                        self.samples.put_nowait(s)
                    except queue.Empty:
                        pass
        finally:
            self.eof = True

    def next(self, timeout: float = 0.1) -> StreamSample | None:
        try:
            s = self.samples.get(timeout=timeout)
            self.samples.task_done()
            return s
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=2)
        if self.thread:
            self.thread.join(timeout=3)
        if p is not None:
            if p.stdout:
                p.stdout.close()
        self.stderr_file.close()


def fitted_prompt(core, count: int, base_prompt: str | None = None) -> str:
    """Change only numbered-observation bounds for shorter windows.

    For original 5s / 10 samples, return the original prompt byte-for-byte.
    This is necessary because the original instructions literally say index 0-9.
    """
    original = base_prompt if base_prompt is not None else core.PROMPT
    if count == 10:
        return original
    assert 2 <= count <= 10
    expected = ('first_index: first observed evidence index, an integer 0-9.',
                'end_index: one PAST the last observed evidence index, integer 1-10.',
                'Use 0 <= first_index < end_index <= 10.')
    assert all(x in original for x in expected), 'Unexpected original Qwen prompt; no silent rewrite'
    return (original.replace(expected[0],
                             f'first_index: first observed evidence index, an integer 0-{count-1}.')
                    .replace(expected[1],
                             f'end_index: one PAST the last observed evidence index, integer 1-{count}.')
                    .replace(expected[2],
                             f'Use 0 <= first_index < end_index <= {count}.'))


def skip(core, journal: dev.LiveJournal, number: int, start: float, end: float,
         reason: str, status: str, frames: list[tuple[int, Any, str, float]] | None = None,
         expected: int = 10):
    frames = frames or []
    journal.add({
        'timeframe': core.interval(start, end), 'window_number': number,
        'window_start_seconds': round(start, 3), 'window_end_seconds': round(end, 3),
        'status': status, 'reason': reason, 'summary': '', 'raw': '', 'seconds': 0,
        'sample_count': len(frames), 'samples_expected': expected,
        'sample_timestamps_seconds': [round(f[3], 4) for f in frames],
        'sample_paths': [f[2] for f in frames],
        'model_processed': False, 'inference_attempted': False,
        'inference_seconds': None, 'queue_delay_seconds': None,
        'response_delay_after_window_seconds': None, 'output_tokens': None,
    })
    print(f'{core.interval(start,end)} | {status} | frames {len(frames)}/{expected} '
          f'| NOT ANALYZED | {reason}', flush=True)


def enqueue_current(core, journal: dev.LiveJournal, jobs: queue.Queue, job: dev.WindowJob,
                    expected: int):
    try:
        jobs.put_nowait(job)
        return
    except queue.Full:
        pass
    # Never delay RTSP reading to wait for an old model request.
    try:
        old = jobs.get_nowait()
    except queue.Empty:
        old = None
    else:
        jobs.task_done()
        if old is not None:
            skip(core, journal, old.number, old.start, old.end,
                 'Replaced an older pending window with newer stream observations',
                 'SKIPPED BACKLOG',
                 [(i, None, p, t) for i, (p,t) in enumerate(zip(old.sample_paths,
                                                             old.capture_times or []))],
                 expected)
    try:
        jobs.put_nowait(job)
    except queue.Full:
        skip(core, journal, job.number, job.start, job.end,
             'Inference queue remains full', 'SKIPPED BACKLOG',
             [(i, None, p, t) for i,(p,t) in enumerate(zip(job.sample_paths,
                                                             job.capture_times or []))], expected)


def export_rtsp(core, folder: Path):
    info = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    audit, actions = dev.read_events(folder / 'events.jsonl')
    duration = float(info.get('elapsed_seconds') or 0)
    core.create_review_book(folder / 'Live_Qwen2B_Review.xlsx',
                            Path('Live_RTSP_Stream.mp4'),
                            {'duration': duration, 'fps': FPS,
                             'width': info['width'], 'height': info['height']},
                            actions, audit)
    summary = dev.export_developer_diagnostics(folder, info, audit)
    # Preserve existing report fields, but add RTSP-specific metrics and accurate
    # moving-window meanings instead of the older webcam hard-coded wording.
    summary.update({
        'source_type': 'RTSP decoded with FFmpeg',
        'rtsp_url_redacted': info['rtsp_url_redacted'],
        'window_seconds': info['window_seconds'],
        'stride_seconds': info['stride_seconds'],
        'samples_per_window': info['samples_per_window'],
        'ffmpeg_frame_size': f"{info['width']}x{info['height']}",
        'ffmpeg_decoder_fps': FPS,
        'ffmpeg_startup_to_first_frame_seconds': info.get('ffmpeg_startup_to_first_frame_seconds'),
        'ffmpeg_frames_decoded': info.get('ffmpeg_frames_decoded'),
        'ffmpeg_reader_queue_drops': info.get('ffmpeg_reader_queue_drops'),
        'ffmpeg_interarrival_seconds_stats': info.get('ffmpeg_interarrival_seconds_stats'),
        'capture_clock_basis': 'monotonic receiver arrival times, relative to first decoded frame',
        'network_latency_seconds': None,
        'network_latency_note': 'Not directly measurable without synchronized source-side timestamps',
        'note': 'Action times are relative to the first decoded RTSP image. Model predictions are '
                'not independently verified. Decode/transport delay is NOT included in response_delay_after_window_seconds.'
    })
    (folder / 'Developer_Session_Summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (folder / 'Developer_Read_Me_First.txt').write_text('\n'.join([
        'HYPERREALITY / LIVE RTSP QWEN2B / DEVELOPER BENCHMARK',
        '====================================================',
        f"Model: {info['model']} (original core SHA256 {info['original_script_sha256'][:12]})",
        f"Source: {info['rtsp_url_redacted']}",
        f"Window/stride: {info['window_seconds']}s / {info['stride_seconds']}s",
        f"Samples: {info['samples_per_window']} at {FPS} fps per analysis request",
        f"Analyzed windows: {summary['analyzed_windows']} / {len(audit)} audited",
        f"Backlog skips: {summary['skipped_backlog_windows']}",
        f"Capture skips: {summary['skipped_capture_windows']}",
        f"Model startup time: {info.get('model_load_seconds')}s",
        f"FFmpeg first-frame delay: {info.get('ffmpeg_startup_to_first_frame_seconds')}s",
        f"FFmpeg reader queue drops: {info.get('ffmpeg_reader_queue_drops')}",
        f"Inference time mean/p95: {summary['inference_time_seconds']['mean']}s / {summary['inference_time_seconds']['p95']}s",
        f"Result delay after observation-window end mean/p95: {summary['response_delay_after_window_seconds']['mean']}s / {summary['response_delay_after_window_seconds']['p95']}s",
        'All timings use receiver clock. Network/RTSP transport latency cannot be directly measured here.',
        'CSV: Developer_Window_Metrics.csv, Developer_Frame_Samples.csv',
        'Audit and model responses: events.jsonl / Live_Qwen2B_Review.xlsx',
        'FFmpeg logs: FFmpeg_RTSP_Decode_Log.txt',
        'These are UNVERIFIED model suggestions, not ground-truth actions.'
    ])+'\n', encoding='utf-8')
    print(f'\nEXCEL: {folder / "Live_Qwen2B_Review.xlsx"}')
    print(f'ACTIONS: {len(actions)} candidates | {len(audit)} audited windows')
    print(f'METRICS: {folder / "Developer_Window_Metrics.csv"}')
    print(f'SUMMARY: {folder / "Developer_Session_Summary.json"}')
    return summary


def run_rtsp(core, url: str, folder: Path, window: int, stride: int, seconds: float,
             preview: bool, record_samples: bool, fake_infer=None,
             first_frame_timeout: float = 15.0, stream_stall_timeout: float = 10.0):
    import cv2
    import numpy as np
    from PIL import Image

    expected = window * FPS
    dev.WINDOW_SECONDS = float(window)
    dev.SAMPLES_PER_WINDOW = expected
    folder.mkdir(parents=True, exist_ok=True)
    image_folder = folder / 'sampled_images'
    image_folder.mkdir(exist_ok=True)
    if fake_infer is None:
        print('Loading offline Qwen3-VL-2B once before connecting to RTSP...', flush=True)
        begun = time.perf_counter()
        model, processor = core.load_qwen()
        model_load = round(time.perf_counter()-begun, 3)
        print(f'Qwen loaded in {model_load:.2f}s', flush=True)

        baseline_prompt = core.PROMPT

        def model_infer(frames):
            # For 10 samples the prompt is completely unchanged. For 2/3-second
            # requests only numeric index bounds change; no instructions changed.
            core.PROMPT = fitted_prompt(core, len(frames), baseline_prompt)
            return core.infer_once(model, processor, frames)
    else:
        model_load = 0.0
        model_infer = fake_infer

    info = {
        'source_name': 'Live_RTSP_Stream.mp4',
        'rtsp_url_redacted': redact_url(url),
        'started_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'model': core.MODEL, 'model_load_seconds': model_load,
        'window_seconds': window, 'stride_seconds': stride,
        'sample_fps': FPS, 'samples_per_window': expected,
        'source_fps': FPS, 'width': DEFAULT_WIDTH, 'height': DEFAULT_HEIGHT,
        'original_script_sha256': dev.EXPECTED_SHA256,
        'elapsed_seconds': 0.0, 'camera_recording_file': None,
        'machine': dev.machine_metadata(),
        'note': 'Receiver wall-clock origin. Genuine network latency unknown. No YOLO.'
    }
    dev.write_session_meta(folder, info)
    journal = dev.LiveJournal(folder, core, info['started_at'])
    jobs = queue.Queue(maxsize=1)
    worker = threading.Thread(target=dev.worker_loop, args=(core, jobs, journal, model_infer),
                              daemon=False, name='qwen-inference-worker')
    reader = FFmpegRTSPReader(url, DEFAULT_WIDTH, DEFAULT_HEIGHT, folder)
    samples: dict[int, tuple[Image.Image, str, float]] = {}
    first_capture = None
    last_capture = None
    next_window = 0
    last_assigned_slot = -1
    skipped_receipt_slots = 0
    saved_samples = 0
    ffmpeg_frames_seen = 0
    sample_time_gaps = []
    latest_frame = None
    recorded = None
    record_path = folder / 'Developer_Sampled_Stream_2fps.mp4'
    sample_csv = (folder / 'Developer_Frame_Samples.csv').open('w', newline='', encoding='utf-8-sig')
    sample_cols = ['sample_slot', 'ffmpeg_decoded_frame_index', 'expected_slot_seconds',
                   'receiver_arrival_seconds', 'arrival_offset_ms', 'received_utc',
                   'image_path', 'width', 'height', 'model_edge_pixels']
    sample_writer = csv.DictWriter(sample_csv, fieldnames=sample_cols)
    sample_writer.writeheader()
    try:
        reader.start()
        worker.start()
        print('RTSP reader started:', redact_url(url), flush=True)
        print(f'Window {window}s | stride {stride}s | {expected} visual observations per request', flush=True)
        print('Q/ESC closes preview; Ctrl+C stops headless sessions.', flush=True)
        if record_samples:
            recorded = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                       FPS, (DEFAULT_WIDTH, DEFAULT_HEIGHT))
            if not recorded.isOpened():
                raise RuntimeError('Could not open the 2fps sampled video recorder')
            print('Sampling-only recording:', record_path)

        while True:
            now = time.perf_counter()
            if first_capture is None and now-reader.started_perf > first_frame_timeout:
                raise RuntimeError(f'No RTSP frames arrived in {first_frame_timeout}s. '
                                   f'Check MediaMTX/FFmpeg publisher and {folder / "FFmpeg_RTSP_Decode_Log.txt"}')
            if last_capture is not None and now - last_capture > stream_stall_timeout:
                print(f'RTSP stalled for {stream_stall_timeout:.0f}s; ending and exporting available results.')
                break
            s = reader.next(timeout=0.08)
            if s is not None:
                ffmpeg_frames_seen += 1
                t = s.received_perf
                if first_capture is None:
                    first_capture = t
                    journal.start_perf = t
                elapsed = max(0.0, t-first_capture)
                if last_capture is not None:
                    sample_time_gaps.append(t-last_capture)
                last_capture = t
                slot = max(0, int(round(elapsed * FPS)))  # Tolerate small decode/arrival jitter
                if slot > last_assigned_slot:
                    if last_assigned_slot >= 0:
                        skipped_receipt_slots += max(0, slot-last_assigned_slot-1)
                    last_assigned_slot = slot
                    array = np.frombuffer(s.pixels, dtype=np.uint8).reshape(
                        (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)).copy()
                    latest_frame = array
                    if recorded:
                        recorded.write(array)
                    image_name = image_folder / f'f{slot+1:06d}.jpg'
                    if not cv2.imwrite(str(image_name), array):
                        raise RuntimeError(f'Failed to write frame: {image_name}')
                    small = cv2.resize(array, (384, 216), interpolation=cv2.INTER_AREA)
                    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                    samples[slot] = pil, str(image_name.resolve()), elapsed
                    saved_samples += 1
                    sample_writer.writerow({
                        'sample_slot': slot, 'ffmpeg_decoded_frame_index': s.number,
                        'expected_slot_seconds': slot/FPS,
                        'receiver_arrival_seconds': round(elapsed, 5),
                        'arrival_offset_ms': round((elapsed-slot/FPS)*1000, 2),
                        'received_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                        'image_path': str(image_name.resolve()),
                        'width': DEFAULT_WIDTH, 'height': DEFAULT_HEIGHT,
                        'model_edge_pixels': core.MODEL_EDGE
                    })
                    sample_csv.flush()

            if first_capture is not None:
                now_elapsed = max(0.0, time.perf_counter()-first_capture)
                # Windows close according to the receiver clock, even when the
                # stream stalls. Never fill missing input slots with copies.
                while now_elapsed >= next_window * stride + window:
                    a = next_window * stride
                    b = a + window
                    indices = range(int(round(a*FPS)), int(round(b*FPS)))
                    present = [(i, samples[i][0], samples[i][1], samples[i][2])
                               for i in indices if i in samples]
                    if len(present) == expected:
                        job = dev.WindowJob(next_window, a, b,
                                            [x[1] for x in present],
                                            [x[2] for x in present], time.perf_counter(),
                                            [x[3] for x in present], jobs.qsize())
                        enqueue_current(core, journal, jobs, job, expected)
                    else:
                        skip(core, journal, next_window, a, b,
                             f'Missing {expected-len(present)} of {expected} RTSP samples',
                             'SKIPPED CAPTURE', present, expected)
                    next_window += 1
                    # Do not discard samples still needed by overlapping windows.
                    keep_from = next_window*stride*FPS
                    samples = {k:v for k,v in samples.items() if k >= keep_from}

            if preview and latest_frame is not None:
                latest, actions, dropped, _ = journal.snapshot()
                dt = time.perf_counter()-(first_capture or time.perf_counter())
                shown = dev.show_overlay(latest_frame, cv2, dt, latest, actions,
                                         dropped, jobs.qsize())
                cv2.imshow('HyperReality LIVE RTSP Qwen2B (Q/ESC to stop)', shown)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                    break

            if seconds > 0 and first_capture is not None and time.perf_counter()-first_capture >= seconds:
                break
            if reader.eof and reader.samples.empty():
                print('RTSP source ended or FFmpeg disconnected. Exporting available windows.')
                break
    except KeyboardInterrupt:
        print('Ctrl+C received; exporting results')
    finally:
        reader.stop()
        if recorded:
            recorded.release()
            info['camera_recording_file'] = str(record_path.resolve())
        sample_csv.close()
        if preview:
            cv2.destroyAllWindows()
        if first_capture is not None and samples:
            elapsed = max(0.0, time.perf_counter()-first_capture)
            start = next_window*stride
            if elapsed > start:
                idx = range(int(start*FPS), int((start+window)*FPS))
                present = [(i, samples[i][0], samples[i][1], samples[i][2]) for i in idx if i in samples]
                skip(core, journal, next_window, start, start+window,
                     'Session stopped before next full analysis window',
                     'INCOMPLETE WINDOW', present, expected)
        if worker.is_alive():
            jobs.put(None)
            jobs.join()
            worker.join()
        duration = max(0.0, (time.perf_counter()-first_capture)) if first_capture else 0.0
        # Preserve desired test length (not slow final exports) when session ends.
        if seconds > 0:
            duration = min(duration, seconds)
        info.update({
            'elapsed_seconds': round(duration, 3),
            'ended_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'ffmpeg_startup_to_first_frame_seconds':
                round(reader.first_frame_perf-reader.started_perf, 3) if reader.first_frame_perf else None,
            'ffmpeg_frames_decoded': reader.frames_decoded,
            'ffmpeg_reader_queue_drops': reader.dropped_in_reader_queue,
            'camera_read_frames_total': ffmpeg_frames_seen,
            'camera_samples_saved_total': saved_samples,
            'camera_samples_missed_slots': skipped_receipt_slots,
            'camera_read_fps_measured': round(ffmpeg_frames_seen/duration, 3) if duration else None,
            'camera_samples_fps_measured': round(saved_samples/duration, 3) if duration else None,
            'camera_read_seconds_stats': None,
            'ffmpeg_interarrival_seconds_stats': dev.metric_stats(sample_time_gaps),
        })
        dev.write_session_meta(folder, info)
        if journal.file.exists():
            export_rtsp(core, folder)
        else:
            print('No valid RTSP observations were captured. See FFmpeg_RTSP_Decode_Log.txt')


def ffmpeg_decode_self_test():
    """Real FFmpeg pipe smoke-test; no RTSP server/network or GPU needed."""
    with tempfile.TemporaryDirectory(prefix='hr_rtsp_ffmpeg_pipe_') as td:
        folder = Path(td)
        output = folder / 'sample.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=10',
                        '-t', '3', '-c:v', 'mpeg4', str(output)], check=True)
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(output),
                   '-an', '-vf', 'fps=2,scale=640:360:force_original_aspect_ratio=decrease,'
                   'pad=640:360:(ow-iw)/2:(oh-ih)/2', '-pix_fmt', 'bgr24',
                   '-f', 'rawvideo', '-']
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw, errors = p.communicate(timeout=15)
        assert p.returncode == 0, errors.decode(errors='replace')
        assert len(raw) == 6*640*360*3, f'Unexpected FFmpeg sample bytes: {len(raw)}'
        print('FFmpeg decoder PASS: 3 seconds -> 6 real BGR24 samples, 640x360')


def self_test(core):
    from PIL import Image
    import zipfile
    ffmpeg_decode_self_test()
    assert core.PROMPT == fitted_prompt(core,10)
    p6 = fitted_prompt(core, 6)
    assert 'integer 0-5' in p6 and 'integer 1-6' in p6 and 'end_index <= 6' in p6
    assert core.PROMPT == dev.original_core().PROMPT
    # Verify variable-window scheduling and overlap without Qwen or an RTSP server.
    for win, stride, expected_starts in [(5,5,[0,5]), (3,3,[0,3,6]), (5,2,[0,2,4,6])]:
        elapsed = 11.5
        starts = [i*stride for i in range(20) if i*stride+win <= elapsed]
        assert starts == expected_starts, (win,stride,starts)
    with tempfile.TemporaryDirectory(prefix='hr_rtsp_benchmark_test_') as td:
        d = Path(td)
        dev.WINDOW_SECONDS = 5.0
        dev.SAMPLES_PER_WINDOW = 10
        images = []
        paths = []
        for i in range(10):
            img = Image.new('RGB',(384,216),(30+i*4,60,80))
            path = d / f'image_{i}.jpg'
            img.save(path)
            images.append(img)
            paths.append(str(path.resolve()))
        info = {
            'rtsp_url_redacted':'rtsp://127.0.0.1:8554/test',
            'source_name': 'Live_RTSP_Stream.mp4',
            'started_at': datetime.now().astimezone().isoformat(timespec='seconds'),
            'original_script_sha256':dev.EXPECTED_SHA256, 'model':core.MODEL,
            'model_load_seconds':0.0, 'width':640, 'height':360,
            'window_seconds':5, 'stride_seconds':2, 'sample_fps':2,
            'samples_per_window':10, 'elapsed_seconds':5,
        }
        dev.write_session_meta(d,info)
        journal = dev.LiveJournal(d,core,info['started_at'])
        journal.start_perf = time.perf_counter()
        job = dev.WindowJob(0,0,5, images,paths,time.perf_counter(),[i/2 for i in range(10)])
        dev.run_one(core,job,lambda frames: (
            'move|wheel|||1|7|The technician moves the visible wheel to the left side.', 20),journal)
        skip(core,journal,1,2,7,'Simulated delayed model','SKIPPED BACKLOG',expected=10)
        result = export_rtsp(core,d)
        assert result['analyzed_windows'] == 1
        assert result['skipped_backlog_windows'] == 1
        assert result['stride_seconds'] == 2
        with zipfile.ZipFile(d/'Live_Qwen2B_Review.xlsx') as z:
            assert z.testzip() is None
        with (d/'Developer_Window_Metrics.csv').open(newline='',encoding='utf-8-sig') as f:
            assert len(list(csv.DictReader(f))) == 2
        assert core.xlsx_values(d/'Live_Qwen2B_Review.xlsx','Actions')[6]['C'] == 'Moved the wheel'
    print('SELF-TEST PASS: unchanged Qwen logic, dynamic sample bounds, 5/5 + 3/3 + 5/2 '
          'scheduling, RTSP frame decoding, model audit and Excel + CSV + JSON exports.')
    print('Note: no actual Qwen inference or RTSP server was used during self-test.')


def main():
    parser = argparse.ArgumentParser(description='HyperReality LIVE RTSP + FFmpeg + Qwen2B developer benchmark')
    parser.add_argument('--rtsp',help='RTSP stream URL, e.g. rtsp://127.0.0.1:8554/hyperreality')
    parser.add_argument('--window',type=int,default=5,choices=[1,2,3,5],help='Context window seconds')
    parser.add_argument('--stride',type=int,default=5,choices=[1,2,3,5],help='New analysis every N seconds (<= window)')
    parser.add_argument('--seconds',type=float,default=60,help='Test duration; 0 means run until stopped')
    parser.add_argument('--no-preview',action='store_true',help='Headless mode; use Ctrl+C to finish')
    parser.add_argument('--record-sampled-video',action='store_true',help='Save 2fps sampled RTSP stream for audit (not full source FPS)')
    parser.add_argument('--test-connection',action='store_true',help='Check RTSP decoding without Qwen and exit')
    parser.add_argument('--self-test',action='store_true',help='Local tests, no RTSP server or Qwen needed')
    parser.add_argument('--output-dir',type=Path,help='Explicit session directory for HTTP-managed jobs')
    parser.add_argument('--rebuild',type=Path,help='Recreate report files from existing session; no Qwen')
    args = parser.parse_args()
    core = dev.original_core()  # Validates unchanged original file SHA256.
    if args.self_test:
        self_test(core)
        return
    if args.rebuild:
        export_rtsp(core,args.rebuild)
        return
    if not args.rtsp:
        parser.error('Provide --rtsp URL (or --self-test)')
    if args.stride > args.window:
        parser.error('--stride must be <= --window')
    if args.test_connection:
        with tempfile.TemporaryDirectory(prefix='hr_rtsp_connection_') as td:
            reader = FFmpegRTSPReader(args.rtsp,DEFAULT_WIDTH,DEFAULT_HEIGHT,Path(td))
            try:
                reader.start()
                began = time.perf_counter()
                got = 0
                while time.perf_counter()-began < 12 and got < 4:
                    if reader.next(0.2) is not None:
                        got += 1
                    elif reader.eof:
                        break
                if got < 4:
                    log = (Path(td)/'FFmpeg_RTSP_Decode_Log.txt').read_text(errors='replace')
                    raise RuntimeError(f'RTSP test got only {got}/4 images; check publisher. {log[-700:]}')
                print(f'RTSP CONNECTION PASS: decoded {got} images at {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}, 2 fps; no Qwen inference')
            finally:
                reader.stop()
        return
    folder = args.output_dir or OUTPUT_BASE / (f'RTSP_{datetime.now():%Y%m%d_%H%M%S}_W{args.window}_S{args.stride}')
    run_rtsp(core,args.rtsp,folder,args.window,args.stride,args.seconds,
             not args.no_preview,args.record_sampled_video)


if __name__=='__main__':
    main()
