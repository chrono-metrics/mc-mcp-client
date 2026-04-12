from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from mc_mcp_client.backends import BaseBackend, LLMBackend, VLLMBackend

RouteHandler = Callable[["MockRequest"], tuple[int, dict[str, str], bytes]]


@dataclass
class MockRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        return json.loads(self.body.decode("utf-8"))


class _MockHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _MockHTTPRequestHandler)
        self.routes: dict[tuple[str, str], RouteHandler] = {}


class _MockHTTPRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _dispatch(self, method: str) -> None:
        route = self.server.routes.get((method, self.path))
        if route is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        request_data = MockRequest(
            method=method,
            path=self.path,
            headers={key: value for key, value in self.headers.items()},
            body=body,
        )
        status, headers, response_body = route(request_data)

        self.send_response(status)
        self.send_header("Content-Length", str(len(response_body)))
        self.send_header("Connection", "close")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(response_body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class MockOpenAIServer:
    def __init__(self) -> None:
        self._server = _MockHTTPServer(("127.0.0.1", 0))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def route(self, method: str, path: str, handler: RouteHandler) -> None:
        self._server.routes[(method, path)] = handler


@pytest.fixture
def mock_openai_server() -> Iterator[MockOpenAIServer]:
    server = MockOpenAIServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_generate_returns_response_text(mock_openai_server) -> None:
    captured_request: dict[str, Any] = {}

    def completions_handler(request: MockRequest) -> tuple[int, dict[str, str], bytes]:
        captured_request["headers"] = request.headers
        captured_request["body"] = request.json()
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthetic answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(body).encode("utf-8")

    mock_openai_server.route("POST", "/v1/chat/completions", completions_handler)

    backend = VLLMBackend(
        model="qwen3-8b",
        base_url=mock_openai_server.base_url,
        api_key="test-key",
        timeout=1.0,
    )
    try:
        result = await backend.generate(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=64,
            temperature=0.2,
            stop=["Observation:"],
        )
    finally:
        if backend._client is not None:
            await backend._client.close()

    assert result == "synthetic answer"
    assert captured_request["headers"]["Authorization"] == "Bearer test-key"
    assert captured_request["body"] == {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "temperature": 0.2,
        "stop": ["Observation:"],
    }


@pytest.mark.asyncio
async def test_check_health_falls_back_to_models(mock_openai_server) -> None:
    def health_handler(_: MockRequest) -> tuple[int, dict[str, str], bytes]:
        return 404, {"Content-Type": "application/json"}, b"{}"

    def models_handler(request: MockRequest) -> tuple[int, dict[str, str], bytes]:
        assert request.headers["Authorization"] == "Bearer test-key"
        body = {"object": "list", "data": [{"id": "qwen3-8b", "object": "model"}]}
        return 200, {"Content-Type": "application/json"}, json.dumps(body).encode("utf-8")

    mock_openai_server.route("GET", "/health", health_handler)
    mock_openai_server.route("GET", "/v1/models", models_handler)

    backend = VLLMBackend(
        model="qwen3-8b",
        base_url=mock_openai_server.base_url,
        api_key="test-key",
        timeout=1.0,
    )

    assert await backend.check_health() is True


@pytest.mark.asyncio
async def test_generate_timeout_raises_timeout_error(mock_openai_server) -> None:
    def slow_completions_handler(_: MockRequest) -> tuple[int, dict[str, str], bytes]:
        time.sleep(0.2)
        body = {
            "id": "chatcmpl-slow",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen3-8b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "too slow"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(body).encode("utf-8")

    mock_openai_server.route("POST", "/v1/chat/completions", slow_completions_handler)

    backend = VLLMBackend(
        model="qwen3-8b",
        base_url=mock_openai_server.base_url,
        api_key="test-key",
        timeout=0.05,
    )
    try:
        with pytest.raises(TimeoutError, match="Timed out waiting for model response"):
            await backend.generate(messages=[{"role": "user", "content": "Hello"}])
    finally:
        if backend._client is not None:
            await backend._client.close()


def test_vllm_backend_implements_async_backend_interface() -> None:
    assert issubclass(VLLMBackend, LLMBackend)
    assert BaseBackend is LLMBackend
