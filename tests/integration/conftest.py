"""
Integration test configuration.

Required environment variables
-------------------------------
MC_MCP_SERVICE_URL   WebSocket URL of the live service.
                     Local:       ws://localhost:9090
                     Cloud Run:   wss://mc-mcp-ws-service-687522919138.us-central1.run.app
                     (omit to skip all live-service tests)

MC_MCP_API_KEY       Bearer token.  Defaults to "mcmcp_dev".

MC_MCP_FAMILY_CONFIG JSON-encoded family config dict for legacy/manual-bootstrap
                     tests that still exercise the explicit override path.
                     Defaults to Zeckendorf: {"mode": "zeckendorf", "depth": 20}

MC_MCP_MODEL_URL     OpenAI-compatible model URL.
                     Example: http://localhost:8080
                     `/v1` is optional; the public VLLMBackend normalizes it.
                     (omit to skip tests that require a real model)

Quick start (local)
-------------------
    export MC_MCP_SERVICE_URL=ws://localhost:9090
    export MC_MCP_API_KEY=mcmcp_dev
    pytest tests/integration/ -v

Via gcloud proxy (recommended for Cloud Run — no public exposure)
-----------------------------------------------------------------
    gcloud run services proxy mc-mcp-ws-service \\
        --project=ladder-gym --region=us-central1 --port=9090
    export MC_MCP_SERVICE_URL=ws://localhost:9090
    pytest tests/integration/ -v
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from urllib import request

import pytest

from mc_mcp_client.backends.base import LLMBackend

# ── Environment ───────────────────────────────────────────────────────────────

_SERVICE_URL = os.getenv("MC_MCP_SERVICE_URL", "")
_API_KEY = os.getenv("MC_MCP_API_KEY", "mcmcp_dev")
_DEFAULT_FAMILY_CONFIG = {"mode": "zeckendorf", "depth": 20}
_MODEL_URL = os.getenv("MC_MCP_MODEL_URL", "")

# ── Pytest markers / hooks ────────────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live_service: requires a live MC-MCP service (set MC_MCP_SERVICE_URL)",
    )
    config.addinivalue_line(
        "markers",
        "requires_model: requires a real LLM model server (set MC_MCP_MODEL_URL)",
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def service_url() -> str:
    if not _SERVICE_URL:
        pytest.skip("MC_MCP_SERVICE_URL not set — skipping live-service tests")
    return _SERVICE_URL


@pytest.fixture(scope="session")
def api_key() -> str:
    return _API_KEY


@pytest.fixture(scope="session")
def family_config() -> dict:
    raw = os.getenv("MC_MCP_FAMILY_CONFIG", "")
    if raw:
        return json.loads(raw)
    return _DEFAULT_FAMILY_CONFIG


@pytest.fixture
def model_url() -> str:
    if not _MODEL_URL:
        pytest.skip("MC_MCP_MODEL_URL not set — skipping model tests")
    return _MODEL_URL


# ── HTTP session helpers ──────────────────────────────────────────────────────


def _http_base(service_url: str) -> str:
    """Convert ws(s):// service URL to http(s):// for REST calls."""
    url = service_url.rstrip("/")
    url = url.replace("wss://", "https://", 1)
    url = url.replace("ws://", "http://", 1)
    return url


def create_session_http(
    service_url: str,
    api_key: str,
    family_config: dict,
    *,
    budget: int = 10,
    synthesis_cadence: int = 4,
) -> dict:
    """POST /v1/sessions and return the parsed response dict."""
    payload = json.dumps(
        {
            "budget": budget,
            "synthesis_cadence": synthesis_cadence,
            "family_config": family_config,
        }
    ).encode()
    req = request.Request(
        f"{_http_base(service_url)}/v1/sessions",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


async def create_session_async(
    service_url: str,
    api_key: str,
    family_config: dict,
    **kwargs,
) -> dict:
    return await asyncio.to_thread(
        create_session_http, service_url, api_key, family_config, **kwargs
    )


def health_check(service_url: str, api_key: str) -> dict:
    """GET /health and return the parsed response dict."""
    req = request.Request(
        f"{_http_base(service_url)}/health",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ── Scripted backend ──────────────────────────────────────────────────────────


class ScriptedBackend(LLMBackend):
    """Deterministic backend that follows a fixed script.

    Responses are consumed in order.  If a response is a callable it is called
    with the current message list and should return a string.
    """

    def __init__(self, responses: list) -> None:
        self._responses: deque = deque(responses)
        self.calls: list[list[dict]] = []

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        del max_tokens, temperature, stop
        self.calls.append([dict(m) for m in messages])
        if not self._responses:
            raise AssertionError(
                f"ScriptedBackend ran out of responses after {len(self.calls)} calls.\n"
                f"Last messages: {messages[-3:]}"
            )
        response = self._responses.popleft()
        if callable(response):
            return response(messages)
        return response

    async def check_health(self) -> bool:
        return True


def synthesis_aware_backend(tool_response: str, stop_response: str) -> ScriptedBackend:
    """Backend that:
    1. Makes one tool call.
    2. Synthesises when synthesis_required arrives.
    3. Stops the episode.

    Handles repeated synthesis_required by producing a synthesis each time.
    """
    call_count = 0

    def respond(messages: list[dict]) -> str:
        nonlocal call_count
        call_count += 1
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if "SYNTHESIS REQUIRED" in last_user:
            return "Conjecture: digit mass is bounded by encoding depth."
        if call_count == 1:
            return tool_response
        return stop_response

    return ScriptedBackend([respond] * 20)
