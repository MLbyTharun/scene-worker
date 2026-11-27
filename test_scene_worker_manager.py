import math
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import scene_worker_manager as manager


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(manager, "CANCELLATIONS_PATH", Path(self.directory.name) / "cancelled.json")
        self.path_patch.start()

    def tearDown(self):
        manager.JOBS.clear()
        manager.CANCELLED_KEYS.clear()
        self.path_patch.stop()
        self.directory.cleanup()

    def test_livekit_token_does_not_leak_into_command_or_metadata(self):
        command, meta = manager.build_command({
            "mode": "livekit", "url": "ws://localhost:7880",
            "token": "private-test-token", "transportTest": True,
        })
        self.assertNotIn("private-test-token", " ".join(command))
        self.assertNotIn("token", meta)
        self.assertIn("--transport-test", command)

    def test_rejects_invalid_livekit_inputs(self):
        for change in (
            {"token": ""}, {"url": "http://localhost"},
            {"window": 1, "stride": 3}, {"seconds": math.nan},
            {"window": 3.5}, {"transportTest": "false"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                manager.build_command({
                    "mode": "livekit", "url": "ws://localhost:7880",
                    "token": "test-token", **change,
                })

    def test_spawn_failure_releases_capacity(self):
        with patch.object(manager.subprocess, "Popen", side_effect=OSError("test spawn failure")):
            with self.assertRaisesRegex(RuntimeError, "Cannot start worker"):
                manager.start_job({"mode": "self-test"})
        self.assertEqual(manager.active_count(), 0)

    def test_retry_returns_the_existing_job_without_spawning_again(self):
        process = MagicMock()
        process.pid = 123
        with patch.object(manager.subprocess, "Popen", return_value=process) as spawn:
            with patch.object(manager.threading, "Thread"):
                first = manager.start_job({"mode": "self-test", "idempotencyKey": "session-one"})
                retry = manager.start_job({"mode": "self-test", "idempotencyKey": "session-one"})
        self.assertEqual(first["id"], retry["id"])
        spawn.assert_called_once()

    def test_cancelled_start_cannot_be_recreated_by_a_delayed_post(self):
        with manager.LOCK:
            manager.remember_cancellation("cancelled-attempt")
        with patch.object(manager.subprocess, "Popen") as spawn:
            with self.assertRaises(manager.CancelledKeyError):
                manager.start_job({"mode": "self-test", "idempotencyKey": "cancelled-attempt"})
        spawn.assert_not_called()
        self.assertIn("cancelled-attempt", manager.CANCELLATIONS_PATH.read_text())

    def test_task_context_is_passed_by_file_and_not_command_line(self):
        context = {"topics": [{"title": "Topic", "tasks": [{"title": "Task", "instruction": "Do work"}]}]}
        process = MagicMock(pid=123)
        with patch.object(manager, "BASE_DIR", Path(self.directory.name)):
            with patch.object(manager.subprocess, "Popen", return_value=process) as spawn:
                with patch.object(manager.threading, "Thread"):
                    request = {
                        "mode": "livekit", "url": "ws://localhost:7880", "token": "test-token",
                        "taskContext": context, "idempotencyKey": "with-context",
                    }
                    job = manager.start_job(request)
                    self.assertEqual(manager.start_job(request)["id"], job["id"])
                    with self.assertRaises(manager.CancelledKeyError):
                        manager.start_job({**request, "taskContext": {"topics": [
                            {"title": "Changed", "tasks": [{"title": "Other", "instruction": ""}]}
                        ]}})
                    spawn.assert_called_once()
        command = spawn.call_args.args[0]
        self.assertNotIn("Do work", " ".join(command))
        self.assertIn("--task-context", command)
        path = Path(job["outputDir"]) / "task-context.json"
        self.assertEqual(json.loads(path.read_text()), context)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_context_file_failure_releases_capacity(self):
        context = {"topics": [{"title": "Topic", "tasks": [{"title": "Task", "instruction": ""}]}]}
        with patch.object(manager, "BASE_DIR", Path(self.directory.name)):
            with patch.object(manager.os, "open", side_effect=OSError("cannot write context")):
                with self.assertRaisesRegex(RuntimeError, "Cannot start worker"):
                    manager.start_job({"mode": "livekit", "url": "ws://localhost:7880",
                                       "token": "test-token", "taskContext": context})
        self.assertEqual(manager.active_count(), 0)

if __name__ == "__main__":
    unittest.main()
