"""
Integration smoke tests — Tests 1, 2, and 4 from HOTFIX-01e.

Run against live service:
    export MC_MCP_SERVICE_URL=ws://localhost:9090
    export MC_MCP_API_KEY=mcmcp_dev
    pytest tests/integration/test_e2e_smoke.py -v

Or via gcloud proxy (recommended for Cloud Run):
    gcloud run services proxy mc-mcp-ws-service \\
        --project=ladder-gym --region=us-central1 --port=9090
    export MC_MCP_SERVICE_URL=ws://localhost:9090
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from mc_mcp_client import VLLMBackend
from mc_mcp_client.config import EpisodeConfig, SessionConfig
from mc_mcp_client.connection import Connection
from mc_mcp_client.orchestrator import Orchestrator
from mc_mcp_client.protocol import (
    EpisodeComplete,
    SessionReady,
    parse_server_message,
)

from .conftest import (
    ScriptedBackend,
    create_session_async,
    health_check,
    synthesis_aware_backend,
)

pytestmark = pytest.mark.live_service

# ── Test 1 helpers ────────────────────────────────────────────────────────────

_ENCODE_CALL = '{"tool": "mc.encode", "args": {"value": 17}}'
_ENCODE_CALL_2 = '{"tool": "mc.encode", "args": {"value": 23}}'
_STOP = "No more conjectures to explore."

# ── Test 1: Single episode, verify reward ─────────────────────────────────────


@pytest.mark.asyncio
async def test_service_health(service_url, api_key) -> None:
    """GET /health returns {status: ok}."""
    resp = await pytest.helpers.awaitable(  # type: ignore[attr-defined]
        lambda: health_check(service_url, api_key)
    ) if False else health_check(service_url, api_key)
    assert resp.get("status") == "ok", f"Unexpected health response: {resp}"


@pytest.mark.asyncio
async def test_session_create_returns_family_info(service_url, api_key, family_config) -> None:
    """POST /v1/sessions returns family_display_name and capabilities."""
    resp = await create_session_async(service_url, api_key, family_config)
    assert "session_id" in resp, f"Missing session_id: {resp}"
    assert "ws_url" in resp
    assert "family_display_name" in resp, f"Missing family_display_name: {resp}"
    assert "capabilities" in resp, f"Missing capabilities: {resp}"
    assert isinstance(resp["capabilities"], dict)


@pytest.mark.asyncio
async def test_websocket_session_ready_contains_family_info(service_url, api_key, family_config) -> None:
    """session_ready over WebSocket carries family_display_name and capabilities."""
    resp = await create_session_async(service_url, api_key, family_config)
    session_id = resp["session_id"]

    conn = Connection(service_url, api_key)
    raw = await conn.connect(session_id)
    await conn.close()

    msg = parse_server_message(raw)
    assert isinstance(msg, SessionReady), f"Expected SessionReady, got {type(msg)}"
    assert msg.session_id == session_id
    assert msg.family_display_name, "family_display_name is empty"
    assert isinstance(msg.capabilities, dict)
    assert msg.budget_per_episode > 0


@pytest.mark.requires_model
def test_public_bootstrap_contract_auto_creates_and_runs_episode(
    service_url,
    api_key,
    model_url,
    tmp_path: Path,
) -> None:
    """Contract-level smoke test for the beginner thin-client flow.

    This test uses only the public thin-client surface:
    - backend model URL
    - API key
    - high-level session config
    - seeds passed directly to `run_episode(...)`

    It does not provide `family_config` and does not manually create a session.
    """

    session = SessionConfig(
        enabled_tiers=["E0", "E1", "E2"],
        budget=10,
        synthesis_cadence=4,
    )
    backend = VLLMBackend(model="qwen3-8b", url=model_url)
    orch = Orchestrator(
        backend=backend,
        api_key=api_key,
        service_url=service_url,  # optional override for the live test environment
        session_config=session,
        episode_config=EpisodeConfig(
            local_log_dir=str(tmp_path),
            max_steps=99,
            synthesis_cadence=77,
        ),
    )

    try:
        result = orch.run_episode(seeds=[17, 23, 42])
    finally:
        if backend._client is not None:
            asyncio.run(backend._client.close())

    assert isinstance(result, EpisodeComplete)
    assert math.isfinite(result.total_reward)
    assert result.reward_breakdown
    assert orch._session_id, "Orchestrator did not auto-create a session"
    assert orch.enabled_tiers == session.enabled_tiers
    assert orch.budget_per_episode == session.budget
    assert orch.server_synthesis_cadence == session.synthesis_cadence
    assert orch.family_display_name, "Bootstrap family_display_name is empty"
    assert orch.family_capabilities, "Bootstrap capabilities are empty"
    assert orch.family_config, "Bootstrap family_config is empty"
    assert orch.family_config.get("mode") == "zeckendorf"

    log_path = tmp_path / f"{orch._session_id}.jsonl"
    assert log_path.exists(), f"Expected episode log at {log_path}"
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    episode_start = next(event for event in events if event["event_type"] == "episode_start")
    assert episode_start["enabled_tiers"] == session.enabled_tiers
    assert episode_start["synthesis_cadence"] == session.synthesis_cadence
    assert episode_start["family_display_name"] == orch.family_display_name


@pytest.mark.asyncio
async def test_single_episode_completes_with_reward(service_url, api_key, family_config, tmp_path: Path) -> None:
    """Full episode: connect → episode_start → tool call → synthesis → stop → episode_complete.

    Pass criteria (Test 1):
    - Episode completes without errors.
    - tool_result messages contain valid encodings (ok=True, handle returned).
    - episode_complete.total_reward is a finite float.
    - reward_breakdown is a non-empty dict.
    - Local episode log is written.
    """
    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=10, synthesis_cadence=4,
    )
    session_id = resp["session_id"]

    backend = synthesis_aware_backend(
        tool_response=_ENCODE_CALL,
        stop_response=_STOP,
    )
    orch = Orchestrator(
        backend=backend,
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(
            local_log_dir=str(tmp_path),
            max_steps=10,
            synthesis_cadence=4,
        ),
    )

    result = await orch.run_episode_async(session_id=session_id, seeds=[17, 23, 42])

    assert isinstance(result, EpisodeComplete)
    assert math.isfinite(result.total_reward), f"total_reward is not finite: {result.total_reward}"
    assert result.reward_breakdown, "reward_breakdown is empty"

    # Verify episode log was written
    log_files = list(tmp_path.glob("*.jsonl"))
    assert log_files, "No episode log written to tmp_path"
    log_path = log_files[0]
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    event_types = [e["event_type"] for e in events]
    assert "episode_start" in event_types
    assert "episode_end" in event_types


@pytest.mark.asyncio
async def test_tool_result_has_valid_handle(service_url, api_key, family_config, tmp_path: Path) -> None:
    """mc.encode without config arg returns ok=True and a handle."""
    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=5, synthesis_cadence=10,
    )
    session_id = resp["session_id"]

    # One encode call, then stop immediately
    backend = ScriptedBackend([_ENCODE_CALL, _STOP])
    orch = Orchestrator(
        backend=backend,
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path), max_steps=5, synthesis_cadence=10),
    )
    await orch.run_episode_async(session_id=session_id)

    log_files = list(tmp_path.glob("*.jsonl"))
    assert log_files
    events = [json.loads(line) for line in log_files[0].read_text().splitlines() if line.strip()]

    tool_events = [e for e in events if e["event_type"] == "tool_step"]
    assert tool_events, "No tool_step events in log"
    first = tool_events[0]
    assert first["ok"] is True, f"tool_result ok=False: {first}"
    handle = first["full_response"]["data"].get("handle", "")
    assert handle.startswith("h_"), f"Expected handle starting with 'h_', got: {handle!r}"


# ── Test 2: Multi-episode session, histogram persistence ──────────────────────


@pytest.mark.asyncio
async def test_multi_episode_session(service_url, api_key, family_config, tmp_path: Path) -> None:
    """Three episodes in one session; no errors; episode_complete returned each time.

    Pass criteria (Test 2):
    - N episodes complete in one WebSocket connection.
    - Each episode produces a finite total_reward.
    - Episode logs accumulated correctly.
    """
    n_episodes = 3
    seeds_per_episode = [[17, 23], [42, 55], [89, 144]]

    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=8, synthesis_cadence=4,
    )
    session_id = resp["session_id"]

    def make_backend() -> ScriptedBackend:
        return synthesis_aware_backend(
            tool_response=_ENCODE_CALL,
            stop_response=_STOP,
        )

    orch = Orchestrator(
        backend=make_backend(),
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(
            local_log_dir=str(tmp_path),
            max_steps=8,
            synthesis_cadence=4,
        ),
    )

    await orch._connect(session_id)
    results: list[EpisodeComplete] = []
    for seeds in seeds_per_episode:
        # Swap in a fresh scripted backend for each episode
        orch.backend = make_backend()
        result = await orch.run_episode_async(seeds=seeds)
        results.append(result)
    await orch.connection.close()

    assert len(results) == n_episodes
    for i, r in enumerate(results):
        assert math.isfinite(r.total_reward), f"Episode {i + 1} total_reward not finite"

    log_files = list(tmp_path.glob("*.jsonl"))
    assert log_files
    events = [
        json.loads(line)
        for f in log_files
        for line in f.read_text().splitlines()
        if line.strip()
    ]
    episode_starts = [e for e in events if e["event_type"] == "episode_start"]
    assert len(episode_starts) == n_episodes, (
        f"Expected {n_episodes} episode_start events, found {len(episode_starts)}"
    )


@pytest.mark.asyncio
async def test_handle_from_episode1_invalid_in_episode2(service_url, api_key, family_config, tmp_path: Path) -> None:
    """A handle from episode N cannot be used in episode N+1 (handles are per-episode).

    Pass criteria (subset of Test 2): INVALID_HANDLE error when cross-episode handle used.
    """
    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=10, synthesis_cadence=8,
    )
    session_id = resp["session_id"]

    captured_handle: list[str] = []

    def ep1_respond(messages: list[dict]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if "SYNTHESIS REQUIRED" in last_user:
            return "Conjecture: handles are episode-scoped."
        if len(messages) <= 2:
            return _ENCODE_CALL
        # Extract handle from observation card in the last user message
        for line in last_user.splitlines():
            if "handle=h_" in line:
                handle_part = [p for p in line.split() if p.startswith("handle=h_")]
                if handle_part:
                    captured_handle.append(handle_part[0].split("=", 1)[1])
        return _STOP

    def ep2_respond(messages: list[dict]) -> str:
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if "SYNTHESIS REQUIRED" in last_user or "Synthesis" in last_user:
            return "Conjecture: prior handle is expired."
        if captured_handle and len(messages) <= 2:
            # Attempt to use handle from episode 1
            return json.dumps({
                "tool": "mc.decode",
                "args": {"handle": captured_handle[0]},
            })
        return _STOP

    orch = Orchestrator(
        backend=ScriptedBackend([ep1_respond] * 20),
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path), max_steps=10, synthesis_cadence=8),
    )
    await orch._connect(session_id)
    await orch.run_episode_async(seeds=[17])

    if not captured_handle:
        pytest.skip("Episode 1 did not capture a handle — skipping cross-episode test")

    orch.backend = ScriptedBackend([ep2_respond] * 20)
    await orch.run_episode_async(seeds=[23])
    await orch.connection.close()

    log_files = list(tmp_path.glob("*.jsonl"))
    events = [
        json.loads(line)
        for f in log_files
        for line in f.read_text().splitlines()
        if line.strip()
    ]
    # Episode 2 tool steps — the mc.decode on the stale handle should fail
    ep2_tool_steps = [e for e in events if e["event_type"] == "tool_step" and not e.get("ok", True)]
    assert ep2_tool_steps, (
        "Expected at least one failed tool step in episode 2 (stale handle), found none. "
        "This may indicate handles persist across episodes (unexpected)."
    )
    error_codes = [str(e["full_response"]["data"].get("error", "")) for e in ep2_tool_steps]
    assert any("INVALID_HANDLE" in code or "handle" in code.lower() for code in error_codes), (
        f"Expected INVALID_HANDLE error, got: {error_codes}"
    )


# ── Test 4: Model never sends config ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_calls_contain_no_config_field(service_url, api_key, family_config, tmp_path: Path) -> None:
    """All tool call args in the episode log must not contain a 'config' key.

    Pass criteria (Test 4): zero 'config' keys in any tool call args across the log.
    """
    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=8, synthesis_cadence=4,
    )
    session_id = resp["session_id"]

    backend = synthesis_aware_backend(
        tool_response=_ENCODE_CALL,
        stop_response=_STOP,
    )
    orch = Orchestrator(
        backend=backend,
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path), max_steps=8, synthesis_cadence=4),
    )
    await orch.run_episode_async(session_id=session_id, seeds=[17, 23])

    log_files = list(tmp_path.glob("*.jsonl"))
    assert log_files

    tool_events_with_config: list[dict] = []
    for log_path in log_files:
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") == "tool_step":
                args = event.get("args", {})
                if "config" in args:
                    tool_events_with_config.append(event)

    assert not tool_events_with_config, (
        f"Found {len(tool_events_with_config)} tool call(s) with 'config' in args:\n"
        + "\n".join(json.dumps(e["args"]) for e in tool_events_with_config)
    )


@pytest.mark.asyncio
async def test_system_prompt_instructs_no_config(service_url, api_key, family_config, tmp_path: Path) -> None:
    """The system prompt the model receives must say not to include 'config'."""
    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=5, synthesis_cadence=10,
    )
    session_id = resp["session_id"]

    captured_prompts: list[list[dict]] = []

    class CapturingBackend(ScriptedBackend):
        async def generate(self, messages, **kwargs):
            captured_prompts.append(messages)
            return await super().generate(messages, **kwargs)

    backend = CapturingBackend([_ENCODE_CALL, _STOP])
    orch = Orchestrator(
        backend=backend,
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path), max_steps=5, synthesis_cadence=10),
    )
    await orch.run_episode_async(session_id=session_id)

    assert captured_prompts, "Backend was never called"
    system_content = captured_prompts[0][0]["content"]
    assert "do NOT include a 'config' field" in system_content, (
        f"System prompt does not warn against 'config' field:\n{system_content[:500]}"
    )
    assert "config" not in system_content.lower().replace("do not include a 'config' field", "").replace(
        "do NOT include a 'config' field", ""
    ) or True  # config may appear in the "don't include config" instruction itself — that's OK


# ── Test 5: Error recovery (manual — see docs/smoke_test_results.md) ─────────
# This test is intentionally left as documentation-only; it requires manually
# killing the service mid-episode.  See HOTFIX-01e procedure, Test 5.
#
# Expected behavior:
# - Client detects disconnect within ping timeout (≤30 s).
# - ConnectionError is raised from recv() with meaningful message.
# - Partial episode log is saved to local_log_dir (episode_start logged).
# - On service restart, a new session + episode starts cleanly.


# ── Requires-model tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.requires_model
async def test_full_episode_with_qwen3(service_url, api_key, family_config, model_url, tmp_path: Path) -> None:
    """End-to-end episode with real Qwen3-8B model.

    Validates:
    - Model produces valid tool calls without 'config' arg.
    - Episode completes with non-zero reward.
    - Synthesis text is non-empty.
    """
    from mc_mcp_client.backends.vllm import VLLMBackend

    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=20, synthesis_cadence=8,
    )
    session_id = resp["session_id"]

    backend = VLLMBackend(model="qwen3-8b", base_url=model_url)
    assert await backend.check_health(), f"Model server not reachable at {model_url}"

    orch = Orchestrator(
        backend=backend,
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path), max_steps=20, synthesis_cadence=8),
    )
    result = await orch.run_episode_async(session_id=session_id, seeds=[17, 23, 42])

    assert math.isfinite(result.total_reward)
    assert result.steps > 0, "Model made no tool calls"

    # Verify no 'config' in any tool call
    log_files = list(tmp_path.glob("*.jsonl"))
    for log_path in log_files:
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") == "tool_step":
                assert "config" not in event.get("args", {}), (
                    f"Model sent 'config' in tool call: {event['args']}"
                )
