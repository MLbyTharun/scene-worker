import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import livekit_scene_worker as worker
from livekit_scene_worker import parse_task_suggestion, task_candidates


class TaskSuggestionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = task_candidates({"topics": [
            {"title": "Topic", "tasks": [
                {"title": "First", "instruction": "One"},
                {"title": "Second", "instruction": "Two"},
            ]}
        ]})

    def test_selects_only_supplied_text_and_no_id(self):
        self.assertEqual(parse_task_suggestion("2", self.candidates), {
            "action": "Second", "title": "Second", "instruction": "Two",
            "topic": "Topic", "status": "UNVERIFIED",
        })

    def test_none_or_invalid_response(self):
        self.assertIsNone(parse_task_suggestion("NONE", self.candidates))
        for value in ("0", "3", "2 or 1", "2\nanything"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_task_suggestion(value, self.candidates)

    def test_inference_uses_task_context_and_journals_unverified_suggestion(self):
        journal = Mock()
        original = worker.core.PROMPT

        def inference(model, processor, frames):
            self.assertIn('"title": "Second"', worker.core.PROMPT)
            self.assertEqual(frames, [(0, "frame")])
            return "2", 1.0

        with patch.object(worker.core, "infer_once", side_effect=inference):
            worker.infer_task_suggestion(None, None, SimpleNamespace(frames=["frame"], number=0), self.candidates, journal)
        self.assertEqual(worker.core.PROMPT, original)
        self.assertEqual(journal.add.call_args.args[0]["status"], "TASK SUGGESTION")
        self.assertEqual(journal.add.call_args.args[1]["instruction"], "Two")

    def test_inference_does_not_emit_action_for_invalid_output(self):
        journal = Mock()
        with patch.object(worker.core, "infer_once", return_value=("not a number", 0.1)):
            worker.infer_task_suggestion(None, None, SimpleNamespace(frames=["frame"], number=0), self.candidates, journal)
        self.assertEqual(journal.add.call_args.args[0]["status"], "REVIEW")
        self.assertIsNone(journal.add.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
