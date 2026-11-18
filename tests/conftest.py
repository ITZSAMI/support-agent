"""Shared fixtures. Tests never call the real Anthropic API or download models."""

import json

import httpx2 as httpx  # the package the Anthropic SDK 2.x uses for HTTP
import pytest
from anthropic import Anthropic, DefaultHttpxClient

from app import metrics
from app.tools import ticket


@pytest.fixture(autouse=True)
def temp_logs(tmp_path, monkeypatch):
    """Write interaction logs and tickets to a temp folder, not logs/."""
    monkeypatch.setattr(metrics, "LOG_PATH", str(tmp_path / "interactions.jsonl"))
    monkeypatch.setattr(ticket, "TICKETS_PATH", str(tmp_path / "tickets.jsonl"))
    return tmp_path


def message(content: list[dict], stop_reason: str) -> dict:
    """A Messages API response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def tool_use(name: str, tool_input: dict, tool_id: str = "toolu_1") -> dict:
    return message(
        [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}], "tool_use"
    )


def text(answer: str) -> dict:
    return message([{"type": "text", "text": answer}], "end_turn")


class FakeAPI:
    """Serves queued Messages API responses through the real Anthropic SDK,
    and records every request body so tests can inspect what was sent."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(200, json=self.responses.pop(0))

    def client(self) -> Anthropic:
        http = DefaultHttpxClient(transport=httpx.MockTransport(self.handler))
        return Anthropic(api_key="test-key", http_client=http, max_retries=0)
