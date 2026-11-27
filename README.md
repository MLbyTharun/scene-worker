# Scene worker: LiveKit, RTSP and no-VR testing

Run all commands below from the backend repository root.

## One-command Docker test with your project's tasks

The local backend must already be running with your project data. No casting
session, camera, manually exported task file, or separate worker/LiveKit terminal
is needed for this test.

One-time setup: copy `scene-worker.env.example` to `scene-worker.env` if it does
not already exist. Set `SCENE_TEST_BACKEND_TOKEN` in that file to your current
login JWT (without `Bearer `), with permission to view this project's assets.
The backend requires authentication; this test does not bypass it.
The env file is ignored by Git and excluded from the Docker build.

The provided defaults are:

```dotenv
SCENE_TEST_BACKEND_URL=http://host.docker.internal:8080/api/authoringv2
SCENE_TEST_PROJECT_ID=6afd4d745b97303aa9215b7f
SCENE_TEST_ORGANIZATION_ID=6afd4d672b7827ebb469a41c
SCENE_TEST_TIMEOUT=600
```

Use the backend gateway on port `8080`, not the browser's port `8082`:
the frontend proxy on `8082` listens only on IPv6 localhost and is not reachable
from Docker. `host.docker.internal` reaches the Mac's gateway from Docker.
These are runtime env settings, not baked into the image; change them without
rebuilding. Never pass the JWT as a build argument.

Increase Docker Desktop memory to at least your planned 10 GB before inference;
Docker uses CPU on Mac and can be much slower than native MPS.
The existing local mirror clock skew still requires `APT_MAX_FUTURE_TIME=86400`
in this Mac's ignored env file; the example keeps the normal 10-second default.
Fix the clock/mirror and restore the default when possible.

Then run just:

```bash
sh workers/scene-worker/test_project_video.sh \
  "/Users/hyperrealitytech/Downloads/IMG_3616.MOV"
```

The script builds the image, starts the Docker manager and isolated test
LiveKit, mounts only that video read-only, fetches every page of your project's
visible topics/tasks through the backend APIs, and sends the context to the
worker at startup. The recording loops as a camera through LiveKit, and the
first processed result is printed in the same terminal. Look for:

- `Loaded ... tasks across ... topics for project ...`
- `LIVEKIT TASK SUGGESTION` with `title` and `instruction`, or an audit reporting
  `NO TASK SUGGESTION` / `REVIEW`. Suggestions are unverified.

Missing IDs, missing/expired authentication and empty task lists fail explicitly;
the project test never silently falls back to task-free inference. This tests
API task retrieval plus the worker/LiveKit pipeline, not the casting lifecycle
or a real glasses app. The existing synthetic transport test remains separate.

The publisher stops its job and exits after the first processed result. Run
the same command again to retest. Docker services remain ready between tests;
stop only this test stack when finished (without deleting data):

```bash
docker compose -p trainxr-scene -f workers/scene-worker/docker-compose.yml \
  --profile project-test stop scene-worker livekit-test
```

## What was integrated

| Handoff mode | Available here | Web/backend integration |
| --- | --- | --- |
| `01_RTSP_Live` | RTSP runner and its shared core | Manager `mode: "rtsp"` |
| `02_Webcam_Live` | Shared webcam runner | Native webcam CLI; a webcam can also be published to LiveKit using the demo below |
| `03_Local_Saved_Videos` | Original core plus the unchanged batch runner | Native/Docker CLI, not a background HTTP upload API |
| New LiveKit adapter | `livekit_scene_worker.py` | Manager `mode: "livekit"` and authenticated backend casting endpoints |

The original `HyperReality_Final_Action_Review.py` model, prompt, parser and
generation settings are unchanged. The RTSP CLI adds an optional `--output-dir`
so HTTP jobs cannot accidentally read another job's results. Diagnostics were
updated for MPS, headless OpenCV and LiveKit.

**Automatic completion mode:** with project tasks, `scene-worker` now uses an
OpenAI-compatible vision API (`VLM_BASE_URL`, `VLM_API_KEY`, `VLM_MODEL`; no
local Qwen weights), sends labelled frames from **all** LiveKit/RTSP streams in
one combined request, and requires consecutive confident `task_completed`
verdicts before emitting `kind="task_completed"`:
`Task completed, next task is : <next task>`. No manual complete button or
device command is used. Completion state is in-memory per worker job; external
review can still override, but suggestions are not external task-system writes.
Legacy `kind="action"` suggestion helpers remain in `livekit_scene_worker.py`
for compatibility but are not the default.

## Communication

```text
Browser/webcam/VR -- video track --> LiveKit room --> Python scene worker
                                      ^
                                      |
                    reliable scene-worker.result data messages

Authenticated client --> backend casting API --> HTTP manager (start/status/stop)
```

LiveKit carries video **and result data** in LiveKit mode. HTTP manages the
worker's lifecycle and exposes its audit/results. RTSP mode is separate and
does not join LiveKit.

Data topic: `scene-worker.result`

```json
{
  "type": "scene-worker.result",
  "jobId": "...",
  "roomName": "...",
  "verified": false,
  "simulation": false,
  "kind": "audit",
  "data": {"status": "NO ACTION", "raw": "NONE", "model_processed": true}
}
```

An action proposal uses `kind: "action"` and the original action fields.
`simulation: true` identifies transport tests that deliberately do not load Qwen.
Skipped/backlogged windows are reported explicitly. The frontend can listen to
LiveKit data packets on this topic; existing frontend code was not migrated.

With task context a successful suggestion instead has this shape:

```json
{
  "type": "scene-worker.result",
  "jobId": "...",
  "roomName": "...",
  "verified": false,
  "simulation": false,
  "kind": "action",
  "data": {
    "action": "Remove wheel",
    "title": "Remove wheel",
    "instruction": "Remove the wheel after loosening the nuts.",
    "topic": "Brake repair",
    "status": "UNVERIFIED"
  }
}
```

The preceding audit event says `TASK SUGGESTION`. `NO TASK SUGGESTION`
means the model returned `NONE`; `REVIEW` means its answer was invalid.
Neither establishes that a task was completed. The client must subscribe to
the LiveKit topic and display the result; publishing does not control glasses.

## Installed environment and model

- Docker image: `scene-worker:local`, Python 3.12, ARM64 Linux CPU PyTorch, FFmpeg,
  Qwen dependencies and LiveKit SDK.
- Native environment: `.venv/`, Python 3.11, Apple Silicon PyTorch with MPS.
  Existing Python was reused; no system interpreter was replaced.
- Model downloaded into `hf-cache/`, shared with Docker through a bind mount.
- Native installed versions recorded in `requirements-mac.lock.txt`.
- Model/cache/output directories are ignored by Git and excluded from image builds.

Native MPS model loading and actual inference were tested on this M1. LiveKit
video -> Qwen -> reliable result data was also tested natively using synthetic
video. This verifies execution/transport, not automotive recognition accuracy.

Docker on Mac cannot access the Apple GPU. Increase Docker Desktop memory from
4 GB to your planned 10 GB before trying real inference. That is a starting
allocation, not a latency/memory guarantee; use one inference process at a time.
The no-model Docker transport test works with the current allocation.

To recreate the native environment:

```bash
sh workers/scene-worker/setup_mac.sh
```

For subsequent native commands, share the model cache:

```bash
export HF_HOME="$PWD/workers/scene-worker/hf-cache"
PY="$PWD/workers/scene-worker/.venv/bin/python"
"$PY" workers/scene-worker/Check_Environment.py --check-model
```

## Easiest no-VR test: Docker, no camera or model inference

The disposable test LiveKit server is isolated from your existing LiveKit
service and has no published host ports.

```bash
docker compose -p trainxr-scene -f workers/scene-worker/docker-compose.yml \
  --profile test run --rm test-publisher
```

Expected output: `LIVEKIT TRANSPORT PASS`, `frames_received >= 2`,
`sample_count: 2`, `simulation: true`, `model_processed: false`.

This publishes moving synthetic video through a real LiveKit room. It verifies
video reception, worker start/stop, and reliable result packets, without VR.
It does **not** claim the model recognized an object.

Stop only this test stack afterward (no volumes are removed):

```bash
docker compose -p trainxr-scene -f workers/scene-worker/docker-compose.yml \
  --profile test stop scene-worker livekit-test
```

## Native no-VR test, including real Qwen

Start a separate local development LiveKit server in terminal 1:

```bash
docker run --rm --name trainxr-scene-demo-livekit \
  -p 127.0.0.1:17980:7880 -p 127.0.0.1:17981:7881 \
  -p 127.0.0.1:17982:17982/udp \
  livekit/livekit-server:v1.9.9 \
  --dev --bind 0.0.0.0 --node-ip 127.0.0.1 --udp-port 17982
```

In terminal 2, start the native manager:

```bash
export HF_HOME="$PWD/workers/scene-worker/hf-cache"
SCENE_WORKER_HOST=127.0.0.1 SCENE_WORKER_PORT=18095 \
  workers/scene-worker/.venv/bin/python workers/scene-worker/scene_worker_manager.py
```

In terminal 3:

```bash
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret
PY="$PWD/workers/scene-worker/.venv/bin/python"

# Fast transport test: no model/camera.
"$PY" workers/scene-worker/test_livekit_transport.py \
  --url ws://127.0.0.1:17980 --manager http://127.0.0.1:18095

# Real inference on synthetic video; NONE is a valid result.
"$PY" workers/scene-worker/test_livekit_transport.py \
  --url ws://127.0.0.1:17980 --manager http://127.0.0.1:18095 \
  --inference --timeout 180

# Substitute a saved recording; replayed as a LiveKit camera.
"$PY" workers/scene-worker/test_livekit_transport.py \
  --url ws://127.0.0.1:17980 --manager http://127.0.0.1:18095 \
  --video "/absolute/path/to/your-recording.mp4" --inference --timeout 180

# Or explicitly opt into your Mac webcam (grant macOS camera permission).
"$PY" workers/scene-worker/test_livekit_transport.py \
  --url ws://127.0.0.1:17980 --manager http://127.0.0.1:18095 \
  --camera 0 --inference --timeout 180
```

The short `devkey`/`secret` keys are only for the isolated development server.
Never use them in production. The demo creates its own room/tokens; a real
browser/VR session must use tokens minted by the backend.
The saved-video LiveKit demo reads at 10 FPS, loops at EOF, and resizes to
320x180; it is a pipeline test, not a source-timing benchmark.
Use Ctrl+C in terminals 1 and 2 to stop the demo services.

## Analyze saved recordings directly (mode 03)

No LiveKit or VR is needed:

```bash
export HF_HOME="$PWD/workers/scene-worker/hf-cache"
cd workers/scene-worker

.venv/bin/python HyperReality_Final_Action_Review.py analyze \
  --video "/absolute/path/to/your-recording.mp4"

.venv/bin/python HyperReality_Qwen2B_Run_Good_Case_Actions.py \
  --videos-dir "/absolute/path/to/your-video-folder"
```

Reports/audits go under `outputs/`. No videos are included in the handoff.
Use recordings you are authorized to process. Never run the original `approve`
command without human review.

## HTTP manager and Docker startup

```bash
docker compose -p trainxr-scene -f workers/scene-worker/docker-compose.yml up
curl --fail http://127.0.0.1:8090/healthz
```

To rebuild:

```bash
docker compose -p trainxr-scene -f workers/scene-worker/docker-compose.yml build scene-worker
```

The local Debian mirror advertised metadata about a day ahead of the build clock.
If this produces `Release file ... is not valid yet`, fix the clock/mirror first.
For this local machine the build was validated with
`build --build-arg APT_MAX_FUTURE_TIME=86400 scene-worker`: a bounded future-date
tolerance, not disabled signature or expiry verification. The default remains 10 seconds.

The manager port is published only on localhost. To require authentication,
copy `scene-worker.env.example` to the ignored `scene-worker.env` and replace
the example token with a real value; send `Authorization: Bearer ...`.
Do not expose the manager directly to the Internet or put worker JWTs/signing
keys in a browser. Browsers use their own backend-issued participant tokens.

Manager endpoints:

- `POST /jobs`: `mode: "livekit"`, `"rtsp"`, `"test-connection"`, or `"self-test"`.
- `GET /jobs/{id}`: status, `latestAction`, `recentEvents`, diagnostic logs.
- `DELETE /jobs/{id}`: stop the job.
- `GET /jobs?key=...`: recover jobs started with an `idempotencyKey`.
- `DELETE /jobs?key=...`: cancel a start even if its delayed POST has not arrived.
- `GET /healthz`: dependency/device diagnostics, not a model-accuracy test.

Room tokens are passed to the LiveKit subprocess through its environment,
not command arguments or public job metadata. Output folders are isolated per
job. Start/status history is in memory; output audits and cancellation tombstones
remain on disk. Cancelled attempt keys cannot be reused; an intentional new start
must use a new attempt key.

RTSP example:

```bash
curl --fail -X POST http://127.0.0.1:8090/jobs \
  -H 'Content-Type: application/json' \
  -d '{"mode":"rtsp","rtspUrl":"rtsp://host.docker.internal:8554/hyperreality","window":3,"stride":3,"seconds":60}'
```

Use `host.docker.internal` only when the RTSP source is on the Mac; for the
optional replay service on this compose network, use `rtsp://mediamtx:8554/...`.
Webcam capture should run natively; Docker Desktop does not provide ordinary
Mac webcam device passthrough.

## Backend integration (opt-in)

These routes belong to the current `authoringv2` casting API, not the legacy
`authoring` service. They require an authenticated JWT with the repository's
`ADMIN` permission, relative to your configured API base/global prefix:

```text
POST   /casting/sessions/:sessionId/scene-worker
GET    /casting/sessions/:sessionId/scene-worker
DELETE /casting/sessions/:sessionId/scene-worker
```

Use an existing live casting session. POST accepts `{}` for the original action
recognition or `{"transportTest":true}` to verify video/result transport without
Qwen. To enable task suggestions, POST both MongoDB IDs:

```json
{
  "projectId": "<project-id>",
  "organizationId": "<organization-id>"
}
```

For example, using your configured API base URL, admin JWT and existing session:

```bash
curl --fail-with-body -X POST "$API_BASE/casting/sessions/$SESSION_ID/scene-worker" \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "x-organization-id: $ORGANIZATION_ID" \
  -H 'Content-Type: application/json' \
  -d "{\"projectId\":\"$PROJECT_ID\",\"organizationId\":\"$ORGANIZATION_ID\"}"
```

The backend checks project/organization scope, fetches every page of the project's
normally visible topics and their tasks (superseded topics are excluded as in
the topics API), then passes topic names, task titles and text instructions to
the worker. Tasks not associated with a topic are not included. Empty task lists
and task context over 3 MiB are rejected. Larger lists may exceed the model's
context or memory limits even below that size. The list is a startup snapshot:
stop and start a new worker to pick up edits. If a start is being retried, changing
the list can conflict with the original attempt; stop it before starting again.

Start is explicit: creating a session does not automatically launch inference.
The regular session-stop lifecycle also cleans up its associated worker.

The context is saved in `outputs/<job-id>/task-context.json` with owner-only
file permissions and persists with the job audit after stopping. Protect and
manage these files as project data. The manager must remain trusted and private.

To try the task prompt with a recording without a backend session, create a JSON
file yourself containing representative, authorized tasks:

```json
{"topics":[{"title":"Brake repair","tasks":[
  {"title":"Loosen wheel nuts","instruction":"Loosen the wheel nuts."},
  {"title":"Remove wheel","instruction":"Remove the wheel."}
]}]}
```

Restart the native manager after code changes, then use the earlier LiveKit
development setup and run:

```bash
workers/scene-worker/.venv/bin/python workers/scene-worker/test_livekit_transport.py \
  --url ws://127.0.0.1:17980 --manager http://127.0.0.1:18095 \
  --video "/absolute/path/to/your-recording.mov" \
  --task-context "/absolute/path/to/your-tasks.json" --inference --timeout 180
```

The terminal shows either `LIVEKIT TASK SUGGESTION` and the title/instruction,
or an audit with `NO TASK SUGGESTION` or `REVIEW`. This demo sends its own JSON
directly to the manager; it does not test the backend's database lookup. To test
that part, start an actual casting session with the backend request above and
subscribe to the same LiveKit room, or inspect the backend worker status route.
The task-aware path has been tested with a recording and illustrative tasks
through native Qwen and LiveKit. This is not a recognition-accuracy validation
for your project, its procedures, or your glasses.

Set these in the backend's environment, then restart only that backend service:

```dotenv
SCENE_WORKER_ENABLED=true
SCENE_WORKER_MANAGER_URL=http://127.0.0.1:8090
SCENE_WORKER_LIVEKIT_URL=ws://host.docker.internal:7880
SCENE_WORKER_TIMEOUT_MS=10000
```

`SCENE_WORKER_LIVEKIT_URL` is the URL reachable **from the Python worker**:
for a native worker use `ws://127.0.0.1:7880`; for a Docker worker joining the
existing host-published LiveKit use `ws://host.docker.internal:7880`; on a shared
Docker network use the actual LiveKit service name. Omit the override to use the
existing backend LiveKit URL. The browser URL is not changed.

Set the same nonempty `SCENE_WORKER_API_TOKEN` in both backend and manager;
the backend requires it. Existing LiveKit signing credentials remain in the backend;
it mints a separate room-scoped subscriber/data-publisher JWT for the worker.
WebRTC ICE/TCP/UDP connectivity must be reachable from the worker as well as the
WebSocket URL. The isolated Docker transport test does not validate an existing
production LiveKit deployment's networking.

Every start reserves a unique attempt key in the casting session. Retries recover
the same job, and stop durably cancels that attempt so a delayed start cannot
recreate it. An intentional later start gets a new key. Leave
`SCENE_WORKER_ENABLED=false` (the default) to preserve the existing no-worker flow.

Backend route/service tests use mocked manager calls. LiveKit transport and
native Qwen execution were tested separately against the real Python manager.
The repository-wide TypeScript check has unrelated pre-existing errors.
