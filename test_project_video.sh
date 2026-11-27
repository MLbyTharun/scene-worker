#!/bin/sh
set -eu

WORKER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$#" -ne 1 ]; then
  echo "Usage: sh workers/scene-worker/test_project_video.sh /absolute/path/to/video.mov" >&2
  exit 1
fi
case "$1" in
  /*) ;;
  *) echo "Use an absolute video path." >&2; exit 1 ;;
esac
if [ ! -f "$1" ] || [ ! -r "$1" ]; then
  echo "Video does not exist or is not readable: $1" >&2
  exit 1
fi
if [ ! -f "$WORKER_DIR/scene-worker.env" ]; then
  echo "Copy scene-worker.env.example to scene-worker.env and set SCENE_TEST_BACKEND_TOKEN first." >&2
  exit 1
fi

export SCENE_TEST_VIDEO="$1"
docker compose --env-file "$WORKER_DIR/scene-worker.env" -p trainxr-scene -f "$WORKER_DIR/docker-compose.yml" \
  --profile project-test build scene-worker
docker compose --env-file "$WORKER_DIR/scene-worker.env" -p trainxr-scene -f "$WORKER_DIR/docker-compose.yml" \
  --profile project-test run --rm --no-deps project-test python -c \
  'import os,sys; sys.exit(0 if os.getenv("SCENE_TEST_BACKEND_TOKEN", "").strip() else "Set SCENE_TEST_BACKEND_TOKEN in workers/scene-worker/scene-worker.env to your current login JWT.")'
docker compose --env-file "$WORKER_DIR/scene-worker.env" -p trainxr-scene -f "$WORKER_DIR/docker-compose.yml" \
  --profile project-test up -d --wait scene-worker livekit-test
exec docker compose --env-file "$WORKER_DIR/scene-worker.env" -p trainxr-scene -f "$WORKER_DIR/docker-compose.yml" \
  --profile project-test run --rm project-test
