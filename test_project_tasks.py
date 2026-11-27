import io
import json
import unittest
import urllib.error
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from test_livekit_transport import load_project_tasks


class ProjectTaskTests(unittest.TestCase):
    project = "6afd4d745b97303aa9215b7f"
    organization = "6afd4d672b7827ebb469a41c"

    def test_fetches_all_pages_with_project_scope_and_authentication(self):
        requests = []
        responses = [
            {"data": [{"_id": "topic-one", "title": "First"}], "meta": {"totalPages": 2}},
            {"data": [{"title": "One", "textInstructions": "Do one"}], "meta": {"totalPages": 2}},
            {"data": [{"title": "Two", "textInstructions": "Do two"}], "meta": {"totalPages": 2}},
            {"data": [{"_id": "topic-two", "title": "Second"}], "meta": {"totalPages": 2}},
            {"data": [{"title": "Three"}], "meta": {"totalPages": 1}},
        ]

        def fetch(request, timeout):
            requests.append(request)
            return io.BytesIO(json.dumps(responses.pop(0)).encode())

        with patch("urllib.request.urlopen", side_effect=fetch):
            context = load_project_tasks("http://backend/api/authoringv2", self.project, self.organization, "test-jwt")
        self.assertEqual(context["topics"], [
            {"title": "First", "tasks": [
                {"title": "One", "instruction": "Do one"},
                {"title": "Two", "instruction": "Do two"},
            ]},
            {"title": "Second", "tasks": [{"title": "Three", "instruction": ""}]},
        ])
        for request in requests:
            query = parse_qs(urlsplit(request.full_url).query)
            self.assertEqual(query["projectId"], [self.project])
            self.assertEqual(query["organizationId"], [self.organization])
            self.assertEqual(request.get_header("Authorization"), "Bearer test-jwt")
        self.assertEqual(parse_qs(urlsplit(requests[2].full_url).query)["page"], ["2"])
        self.assertEqual(parse_qs(urlsplit(requests[4].full_url).query)["topicId"], ["topic-two"])

    def test_missing_project_or_token_fails_before_network_request(self):
        with patch("urllib.request.urlopen") as fetch:
            for project, token in (("", "test-jwt"), (self.project, "")):
                with self.assertRaises(ValueError):
                    load_project_tasks("http://backend/api/authoringv2", project, self.organization, token)
            fetch.assert_not_called()

    def test_auth_failure_and_empty_project_are_not_successes(self):
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            "http://backend", 401, "Unauthorized", {}, None,
        )):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                load_project_tasks("http://backend/api/authoringv2", self.project, self.organization, "test-jwt")
        with patch("urllib.request.urlopen", return_value=io.BytesIO(
            b'{"data":[],"meta":{"totalPages":1}}',
        )):
            with self.assertRaisesRegex(ValueError, "No tasks found"):
                load_project_tasks("http://backend/api/authoringv2", self.project, self.organization, "test-jwt")


if __name__ == "__main__":
    unittest.main()
