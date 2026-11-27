"""Publish test video, a saved recording, or a webcam through LiveKit without VR."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

import numpy as np
from livekit import api, rtc


def load_project_tasks(base_url: str, project_id: str, organization_id: str, token: str) -> dict:
    for name, value in (("project ID", project_id), ("organization ID", organization_id)):
        if not re.fullmatch(r"[a-fA-F0-9]{24}", value):
            raise ValueError(f"A valid {name} is required for --project-tasks")
    if not token.strip():
        raise ValueError("Set SCENE_TEST_BACKEND_TOKEN in scene-worker.env to your current login JWT")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("SCENE_TEST_BACKEND_URL must be an HTTP(S) authoringv2 API base URL")

    def pages(resource: str, **filters):
        page = 1
        while True:
            query = urllib.parse.urlencode({
                "projectId": project_id, "organizationId": organization_id,
                "limit": 100, "page": page, **filters,
            })
            request = urllib.request.Request(
                f"{base_url.rstrip('/')}/{resource}?{query}",
                headers={"Authorization": f"Bearer {token.strip()}",
                         "x-organization-id": organization_id},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    result = json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(
                    f"Cannot fetch {resource}: HTTP {exc.code}. "
                    "Check backend URL, login token, project and organization permissions."
                ) from None
            except urllib.error.URLError:
                raise RuntimeError("Cannot reach backend; start it and check SCENE_TEST_BACKEND_URL") from None
            if not isinstance(result, dict) or not isinstance(result.get("data"), list):
                raise ValueError(f"Invalid {resource} API response: expected data and meta.totalPages")
            total = result.get("meta", {}).get("totalPages")
            if type(total) is not int or total < 1:
                raise ValueError(f"Invalid pagination in {resource} API response")
            yield from result["data"]
            if page >= total:
                return
            page += 1

    topics = []
    for topic in pages("topics"):
        topic_id = topic.get("_id") or topic.get("topicId")
        if not topic_id:
            raise ValueError("Topic API response is missing its ID")
        tasks = [
            {"title": str(task.get("title") or ""),
             "instruction": str(task.get("textInstructions") or "")}
            for task in pages("tasks", topicId=topic_id)
        ]
        topics.append({"title": str(topic.get("title") or topic.get("name") or ""), "tasks": tasks})
    context = {"projectId": project_id, "organizationId": organization_id, "topics": topics}
    count = sum(len(topic["tasks"]) for topic in topics)
    if not count:
        raise ValueError("No tasks found for this project and organization")
    if len(json.dumps(context, ensure_ascii=False).encode("utf-8")) > 3 * 1024 * 1024:
        raise ValueError("Project task context exceeds 3 MiB")
    print(f"Loaded {count} tasks across {len(topics)} topics for project {project_id}", flush=True)
    return context


async def run(args: argparse.Namespace) -> None:
    task_context = None
    if args.project_tasks:
        if not args.inference:
            raise ValueError("--project-tasks requires --inference")
        task_context = await asyncio.to_thread(
            load_project_tasks,
            os.getenv("SCENE_TEST_BACKEND_URL", "http://host.docker.internal:8080/api/authoringv2"),
            os.getenv("SCENE_TEST_PROJECT_ID", ""),
            os.getenv("SCENE_TEST_ORGANIZATION_ID", ""),
            os.getenv("SCENE_TEST_BACKEND_TOKEN", ""),
        )
    room_name = f"scene-test-{uuid.uuid4().hex[:10]}"
    key = os.environ["LIVEKIT_API_KEY"]
    secret = os.environ["LIVEKIT_API_SECRET"]

    def token(identity: str, publish: bool) -> str:
        return api.AccessToken(key, secret).with_identity(identity).with_grants(
            api.VideoGrants(
                room_join=True, room=room_name, can_publish=publish,
                can_subscribe=True, can_publish_data=True,
            )
        ).to_jwt()

    def request(method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.manager + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        manager_token = os.environ.get("SCENE_WORKER_API_TOKEN", "")
        if manager_token:
            req.add_header("Authorization", f"Bearer {manager_token}")
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    room = rtc.Room()
    result: asyncio.Future = asyncio.get_running_loop().create_future()
    job_id = None
    source = rtc.VideoSource(320, 180)
    capture = None
    if args.task_context:
        if not args.inference:
            raise ValueError("--task-context requires --inference")
        with open(args.task_context, encoding="utf-8") as handle:
            task_context = json.load(handle)
    if args.video or args.camera is not None:
        import cv2
        capture = cv2.VideoCapture(args.video if args.video else args.camera)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Cannot open the specified video/camera")

    @room.on("data_received")
    def data_received(packet):
        if packet.topic != "scene-worker.result":
            return
        event = json.loads(packet.data)
        if event.get("jobId") != job_id:
            return
        if args.inference and event.get("data", {}).get("status", "").startswith("SKIPPED"):
            return
        if task_context and event.get("kind") == "audit" and event.get("data", {}).get("status") == "TASK SUGGESTION":
            return
        if not result.done():
            result.set_result(event)

    try:
        await room.connect(args.url, token("test-camera", True))
        track = rtc.LocalVideoTrack.create_video_track("test-camera", source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await room.local_participant.publish_track(track, options)
        job = await asyncio.to_thread(request, "POST", "/jobs", {
            "idempotencyKey": room_name,
            "mode": "livekit", "url": args.worker_url or args.url,
            "token": token("scene-worker-test", False), "roomName": room_name,
            "participantIdentity": "test-camera", "window": 1, "stride": 1,
            "seconds": args.timeout + 15, "transportTest": not args.inference,
            **({"taskContext": task_context} if task_context is not None else {}),
        })
        job_id = job["id"]
        for number in range(args.timeout * 10):
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            if capture is not None:
                ok, frame = capture.read()
                if not ok and args.video:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Video/camera stopped supplying frames")
                image = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2RGB)
            else:
                image[:, :, 2] = 60
                left = number % 240
                image[40:120, left:left + 80] = (230, 180, 30)
            source.capture_frame(rtc.VideoFrame(320, 180, rtc.VideoBufferType.RGB24, image.tobytes()))
            await asyncio.sleep(0.1)
            if result.done():
                event = result.result()
                assert event["verified"] is False and event["simulation"] is not args.inference, event
                if event["kind"] == "action":
                    assert task_context is not None and event["data"]["status"] == "UNVERIFIED", event
                    label = "LIVEKIT TASK SUGGESTION"
                else:
                    assert event["kind"] == "audit", event
                    assert event["data"]["model_processed"] is args.inference, event
                    if not args.inference:
                        assert event["data"]["frames_received"] >= 2, event
                    assert event["data"]["sample_count"] == 2, event
                    label = "LIVEKIT INFERENCE PASS" if args.inference else "LIVEKIT TRANSPORT PASS"
                print(f"{label}:", json.dumps(event))
                return
        status = await asyncio.to_thread(request, "GET", f"/jobs/{job_id}")
        raise RuntimeError(f"No worker data result in {args.timeout} seconds: {status}")
    finally:
        if capture is not None:
            capture.release()
        if job_id:
            await asyncio.to_thread(request, "DELETE", f"/jobs/{job_id}")
        await source.aclose()
        await room.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:7880")
    parser.add_argument("--worker-url", help="LiveKit URL reachable from the worker container")
    parser.add_argument("--manager", default="http://127.0.0.1:8090")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video", help="Replay an authorized recording as a LiveKit camera")
    inputs.add_argument("--camera", type=int, help="Explicitly open this webcam index")
    parser.add_argument("--inference", action="store_true", help="Load Qwen; not just a transport test")
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--task-context", help="JSON file containing topics and tasks for inference")
    context.add_argument("--project-tasks", action="store_true", help="Fetch project tasks using SCENE_TEST_* environment settings")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("SCENE_TEST_TIMEOUT", "25")), help="Use 180 or more for real inference")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        asyncio.run(run(args))
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Test failed: {exc}\n")


if __name__ == "__main__":
    main()
