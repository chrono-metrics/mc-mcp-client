from __future__ import annotations

import inspect
import json
from collections import deque
from pathlib import Path

import pytest
from websockets.server import ServerConnection

from mc_mcp_client.backends.base import LLMBackend
from mc_mcp_client.config import DEFAULT_SERVICE_URL, EpisodeConfig, SessionConfig
from mc_mcp_client.orchestrator import Orchestrator, _ToolBatch
from mc_mcp_client.protocol import EpisodeComplete, EpisodeEnd, Synthesis, ToolCall


class ScriptedBackend(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        self._responses = deque(responses)
        self.calls: list[list[dict]] = []

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        del max_tokens, temperature, stop
        self.calls.append([dict(message) for message in messages])
        if not self._responses:
            raise AssertionError("No scripted responses remaining.")
        return self._responses.popleft()

    async def check_health(self) -> bool:
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────


def _episode_ready(
    session_id: str,
    *,
    episode_number: int = 1,
    seeds: list[int] | None = None,
    budget: int = 40,
    prior_conjectures: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "type": "episode_ready",
            "id": f"ep_{episode_number}",
            "episode_id": f"{session_id}_ep{episode_number}",
            "episode_number": episode_number,
            "seeds": seeds or [],
            "budget": budget,
            "prior_conjectures": prior_conjectures or [],
        }
    )


async def _recv_episode_start(connection: ServerConnection) -> dict:
    """Receive and return an episode_start message from the client."""
    raw = await connection.recv()
    msg = json.loads(raw)
    assert msg["type"] == "episode_start", f"Expected episode_start, got {msg['type']!r}"
    return msg


def _batch_response(*payloads: dict) -> str:
    return "\n".join(json.dumps(payload) for payload in payloads)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_episode_handles_tool_synthesis_and_stop(mock_websocket_server, tmp_path: Path) -> None:
    session_id = "episode-happy-path"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={
            "budget_per_episode": 2,
            "synthesis_cadence": 1,
            "enabled_tiers": ["E0", "E1", "E2"],
            "family_display_name": "Zeckendorf",
            "capabilities": {"arithmetic": True, "zoom": False},
        },
    )
    received_messages: list[dict] = []

    async def episode_handler(connection: ServerConnection, _: str) -> None:
        ep_start = await _recv_episode_start(connection)
        received_messages.append(ep_start)
        await connection.send(
            _episode_ready(session_id, episode_number=1, seeds=[5, 8, 13], budget=2)
        )

        tool_call = json.loads(await connection.recv())
        received_messages.append(tool_call)
        await connection.send(
            json.dumps(
                {
                    "type": "tool_result",
                    "id": tool_call["id"],
                    "ok": True,
                    "data": {"handle": "h_seed", "summary": {"depth_used": 8, "family": "zeckendorf"}},
                    "step": 1,
                    "budget_remaining": 1,
                    "reward_so_far": 0.25,
                    "reward_multiplier": 1.0,
                }
            )
        )
        await connection.send(
            json.dumps(
                {
                    "type": "synthesis_required",
                    "step": 1,
                    "budget_remaining": 1,
                    "reward_so_far": 0.25,
                    "reward_multiplier": 1.0,
                }
            )
        )

        synthesis = json.loads(await connection.recv())
        received_messages.append(synthesis)
        await connection.send(
            json.dumps(
                {
                    "type": "synthesis_scored",
                    "id": synthesis["id"],
                    "reward_after": 1.25,
                    "reward_delta": 1.0,
                    "conjectures_extracted": 1,
                    "conjecture_ids": ["conj_1234"],
                    "prior_relevant": [],
                    "reward_multiplier": 1.0,
                }
            )
        )

        episode_end = json.loads(await connection.recv())
        received_messages.append(episode_end)
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 1.25,
                    "reward_breakdown": {"reward_base": 1.25, "synthesis_skip_penalty": 0.0},
                    "steps": 1,
                    "syntheses": 1,
                    "conjectures_produced": 1,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, episode_handler)

    backend = ScriptedBackend(
        [
            '{"tool": "mc.encode", "args": {"value": 5}}',
            "Conjecture: digit mass appears monotone across the seeded values.",
            "No more conjectures to explore.",
        ]
    )
    orchestrator = Orchestrator(
        backend=backend,
        service_url=mock_websocket_server.service_url,
        api_key=mock_websocket_server.api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path), synthesis_cadence=1, max_steps=2),
    )

    result = await orchestrator.run_episode_async(session_id, seeds=[5, 8, 13])

    assert result.total_reward == pytest.approx(1.25)
    assert result.steps == 1
    # received_messages: [episode_start, tool_call, synthesis, episode_end]
    assert received_messages[0]["type"] == "episode_start"
    assert received_messages[1]["type"] == "tool_call"
    assert received_messages[1]["tool"] == "mc.encode"
    assert received_messages[2]["type"] == "synthesis"
    assert "Conjecture:" in received_messages[2]["text"]
    assert received_messages[3] == {"type": "episode_end", "reason": "no_more_conjectures"}

    first_prompt = backend.calls[0]
    assert first_prompt[0]["role"] == "system"
    assert "Zeckendorf" in first_prompt[0]["content"]
    assert "do NOT include a 'config' field" in first_prompt[0]["content"]
    assert "Prefer exactly one tool call per turn." in first_prompt[0]["content"]
    assert "Default format: output exactly one JSON object and nothing else" in first_prompt[0]["content"]
    assert "Same-turn batches are allowed only when every tool is one of" in first_prompt[0]["content"]
    assert "`mc.hist_calibrate`" in first_prompt[0]["content"]
    assert "`mc.compare` and `mc.arithmetic` operate on handles" in first_prompt[0]["content"]
    assert "Seed integers: 5, 8, 13" in first_prompt[1]["content"]
    assert "prefer one tool call as a single JSON object" in first_prompt[1]["content"]
    assert "any batch containing another tool is invalid" in first_prompt[1]["content"]

    log_path = tmp_path / f"{session_id}.jsonl"
    assert log_path.exists()
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [event["event_type"] for event in events] == [
        "episode_start",
        "tool_step",
        "synthesis",
        "episode_end",
    ]
    assert events[1]["observation_card"].startswith("tool=mc.encode")
    assert events[2]["synthesis_text"].startswith("Conjecture:")
    assert events[3]["stopped_reason"] == "no_more_conjectures"


@pytest.mark.asyncio
async def test_run_episode_retries_once_after_parse_failure(mock_websocket_server, tmp_path: Path) -> None:
    session_id = "episode-parse-retry"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={"budget_per_episode": 3, "synthesis_cadence": 8},
    )
    received_messages: list[dict] = []

    async def parse_retry_handler(connection: ServerConnection, _: str) -> None:
        await _recv_episode_start(connection)
        await connection.send(
            _episode_ready(session_id, episode_number=1, seeds=[], budget=3)
        )

        tool_call = json.loads(await connection.recv())
        received_messages.append(tool_call)
        await connection.send(
            json.dumps(
                {
                    "type": "tool_result",
                    "id": tool_call["id"],
                    "ok": False,
                    "data": {"error": "INVALID_HANDLE", "detail": "Handle may have expired."},
                    "step": 1,
                    "budget_remaining": 2,
                    "reward_so_far": 0.0,
                    "reward_multiplier": 1.0,
                }
            )
        )

        episode_end = json.loads(await connection.recv())
        received_messages.append(episode_end)
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 0.0,
                    "reward_breakdown": {"reward_base": 0.0, "synthesis_skip_penalty": 0.0},
                    "steps": 1,
                    "syntheses": 0,
                    "conjectures_produced": 0,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, parse_retry_handler)

    backend = ScriptedBackend(
        [
            "I should inspect the handle situation first.",
            "```json\n{\"tool\": \"mc.inspect\", \"args\": {\"handle\": \"h_dead\", \"view\": \"prefix\"}}\n```",
            "I have no further conjectures",
        ]
    )
    orchestrator = Orchestrator(
        backend=backend,
        service_url=mock_websocket_server.service_url,
        api_key=mock_websocket_server.api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    result = await orchestrator.run_episode_async(session_id)

    assert result.steps == 1
    assert received_messages[0]["tool"] == "mc.inspect"
    assert received_messages[0]["args"] == {"handle": "h_dead", "view": "prefix"}
    assert received_messages[1] == {"type": "episode_end", "reason": "no_more_conjectures"}
    assert any(
        message["role"] == "user" and "If you must batch in the same turn" in message["content"]
        for message in backend.calls[1]
    )
    assert any(
        message["role"] == "user" and "Do not include prose, markdown fences, or a `config` field." in message["content"]
        for message in backend.calls[1]
    )


@pytest.mark.asyncio
async def test_run_episode_executes_batchable_tool_batch_and_concatenates_observations(
    mock_websocket_server,
    tmp_path: Path,
) -> None:
    session_id = "episode-batchable-tools"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={"budget_per_episode": 5, "synthesis_cadence": 8},
    )
    received_messages: list[dict] = []

    async def batch_handler(connection: ServerConnection, _: str) -> None:
        await _recv_episode_start(connection)
        await connection.send(
            _episode_ready(session_id, episode_number=1, seeds=[17, 23, 42], budget=5)
        )

        for step, value in enumerate([17, 23, 42], start=1):
            tool_call = json.loads(await connection.recv())
            received_messages.append(tool_call)
            assert tool_call["type"] == "tool_call"
            assert tool_call["tool"] == "mc.encode"
            assert tool_call["args"] == {"value": value}
            await connection.send(
                json.dumps(
                    {
                        "type": "tool_result",
                        "id": tool_call["id"],
                        "ok": True,
                        "data": {"handle": f"h_{value}"},
                        "step": step,
                        "budget_remaining": 5 - step,
                        "reward_so_far": float(step),
                        "reward_multiplier": 1.0,
                    }
                )
            )

        episode_end = json.loads(await connection.recv())
        received_messages.append(episode_end)
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 3.0,
                    "reward_breakdown": {"reward_base": 3.0},
                    "steps": 3,
                    "syntheses": 0,
                    "conjectures_produced": 0,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, batch_handler)

    backend = ScriptedBackend(
        [
            _batch_response(
                {"tool": "mc.encode", "args": {"value": 17}},
                {"tool": "mc.encode", "args": {"value": 23}},
                {"tool": "mc.encode", "args": {"value": 42}},
            ),
            "No more conjectures to explore.",
        ]
    )
    orchestrator = Orchestrator(
        backend=backend,
        service_url=mock_websocket_server.service_url,
        api_key=mock_websocket_server.api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    result = await orchestrator.run_episode_async(session_id, seeds=[17, 23, 42])

    assert result.steps == 3
    assert received_messages[-1] == {"type": "episode_end", "reason": "no_more_conjectures"}
    assert [message["tool"] for message in received_messages[:-1]] == ["mc.encode", "mc.encode", "mc.encode"]

    observation_messages = [
        message["content"]
        for message in backend.calls[1]
        if message["role"] == "user" and "tool=mc.encode" in message["content"]
    ]
    assert len(observation_messages) == 1
    assert observation_messages[0].count("tool=mc.encode") == 3
    assert "\n\n" in observation_messages[0]

    log_path = tmp_path / f"{session_id}.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tool_steps = [event for event in events if event["event_type"] == "tool_step"]
    assert len(tool_steps) == 3


@pytest.mark.asyncio
async def test_run_episode_rejects_mixed_tool_batch_and_requests_retry(
    mock_websocket_server,
    tmp_path: Path,
) -> None:
    session_id = "episode-mixed-batch"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={"budget_per_episode": 5, "synthesis_cadence": 8},
    )
    received_messages: list[dict] = []

    async def mixed_batch_handler(connection: ServerConnection, _: str) -> None:
        await _recv_episode_start(connection)
        await connection.send(
            _episode_ready(session_id, episode_number=1, seeds=[17, 23], budget=5)
        )

        episode_end = json.loads(await connection.recv())
        received_messages.append(episode_end)
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 0.0,
                    "reward_breakdown": {"reward_base": 0.0},
                    "steps": 0,
                    "syntheses": 0,
                    "conjectures_produced": 0,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, mixed_batch_handler)

    backend = ScriptedBackend(
        [
            _batch_response(
                {"tool": "mc.encode", "args": {"value": 17}},
                {"tool": "mc.compare", "args": {"lhs": "h_left", "rhs": "h_right"}},
            ),
            "No more conjectures to explore.",
        ]
    )
    orchestrator = Orchestrator(
        backend=backend,
        service_url=mock_websocket_server.service_url,
        api_key=mock_websocket_server.api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    result = await orchestrator.run_episode_async(session_id, seeds=[17, 23])

    assert result.steps == 0
    assert received_messages == [{"type": "episode_end", "reason": "no_more_conjectures"}]
    assert any(
        message["role"] == "user" and "If you must batch in the same turn" in message["content"]
        for message in backend.calls[1]
    )

    log_path = tmp_path / f"{session_id}.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [event["event_type"] for event in events] == ["episode_start", "episode_end"]


@pytest.mark.asyncio
async def test_run_episode_stops_batch_drain_when_synthesis_required_mid_batch(
    mock_websocket_server,
    tmp_path: Path,
) -> None:
    session_id = "episode-batch-synthesis"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={"budget_per_episode": 5, "synthesis_cadence": 2},
    )
    received_messages: list[dict] = []

    async def batch_synthesis_handler(connection: ServerConnection, _: str) -> None:
        await _recv_episode_start(connection)
        await connection.send(
            _episode_ready(session_id, episode_number=1, seeds=[17, 23, 42], budget=5)
        )

        for step, value in enumerate([17, 23], start=1):
            tool_call = json.loads(await connection.recv())
            received_messages.append(tool_call)
            await connection.send(
                json.dumps(
                    {
                        "type": "tool_result",
                        "id": tool_call["id"],
                        "ok": True,
                        "data": {"handle": f"h_{value}"},
                        "step": step,
                        "budget_remaining": 5 - step,
                        "reward_so_far": step / 2,
                        "reward_multiplier": 1.0,
                    }
                )
            )

        await connection.send(
            json.dumps(
                {
                    "type": "synthesis_required",
                    "step": 2,
                    "budget_remaining": 3,
                    "reward_so_far": 1.0,
                    "reward_multiplier": 1.0,
                }
            )
        )

        synthesis = json.loads(await connection.recv())
        received_messages.append(synthesis)
        await connection.send(
            json.dumps(
                {
                    "type": "synthesis_scored",
                    "id": synthesis["id"],
                    "reward_after": 2.0,
                    "reward_delta": 1.0,
                    "conjectures_extracted": 1,
                    "conjecture_ids": ["conj_mid_batch"],
                    "prior_relevant": [],
                    "reward_multiplier": 1.0,
                }
            )
        )

        episode_end = json.loads(await connection.recv())
        received_messages.append(episode_end)
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 2.0,
                    "reward_breakdown": {"reward_base": 2.0},
                    "steps": 2,
                    "syntheses": 1,
                    "conjectures_produced": 1,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, batch_synthesis_handler)

    backend = ScriptedBackend(
        [
            _batch_response(
                {"tool": "mc.encode", "args": {"value": 17}},
                {"tool": "mc.encode", "args": {"value": 23}},
                {"tool": "mc.encode", "args": {"value": 42}},
            ),
            "Conjecture: the handles preserve input order across the seed list.",
            "No more conjectures to explore.",
        ]
    )
    orchestrator = Orchestrator(
        backend=backend,
        service_url=mock_websocket_server.service_url,
        api_key=mock_websocket_server.api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path), synthesis_cadence=2, max_steps=5),
    )

    result = await orchestrator.run_episode_async(session_id, seeds=[17, 23, 42])

    assert result.steps == 2
    assert [message["tool"] for message in received_messages[:2]] == ["mc.encode", "mc.encode"]
    assert received_messages[2]["type"] == "synthesis"
    assert received_messages[3] == {"type": "episode_end", "reason": "no_more_conjectures"}

    observation_messages = [
        message["content"]
        for message in backend.calls[1]
        if message["role"] == "user" and "tool=mc.encode" in message["content"]
    ]
    assert len(observation_messages) == 1
    assert observation_messages[0].count("tool=mc.encode") == 2


@pytest.mark.asyncio
async def test_run_session_runs_multiple_episodes(mock_websocket_server, tmp_path: Path) -> None:
    session_id = "multi-episode-session"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={
            "budget_per_episode": 5,
            "synthesis_cadence": 8,
            "family_display_name": "Fibonacci",
        },
    )

    async def multi_episode_handler(connection: ServerConnection, _: str) -> None:
        # Episode 1
        ep_start_1 = await _recv_episode_start(connection)
        assert ep_start_1["seeds"] == [17, 23]
        await connection.send(_episode_ready(session_id, episode_number=1, seeds=[17, 23], budget=5))
        ep_end_1 = json.loads(await connection.recv())
        assert ep_end_1["type"] == "episode_end"
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 0.5,
                    "reward_breakdown": {},
                    "steps": 0,
                    "syntheses": 0,
                    "conjectures_produced": 0,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        # Episode 2
        ep_start_2 = await _recv_episode_start(connection)
        assert ep_start_2["seeds"] == [42, 55]
        await connection.send(_episode_ready(session_id, episode_number=2, seeds=[42, 55], budget=5))
        ep_end_2 = json.loads(await connection.recv())
        assert ep_end_2["type"] == "episode_end"
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 1.0,
                    "reward_breakdown": {},
                    "steps": 0,
                    "syntheses": 0,
                    "conjectures_produced": 0,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        await connection.wait_closed()

    # Single handler covers both episodes on the same connection
    mock_websocket_server.queue_handler(session_id, multi_episode_handler)

    backend = ScriptedBackend(
        [
            "No more conjectures to explore.",  # episode 1
            "No more conjectures to explore.",  # episode 2
        ]
    )
    orchestrator = Orchestrator(
        backend=backend,
        service_url=mock_websocket_server.service_url,
        api_key=mock_websocket_server.api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    # Manually connect (simulating run_session's _connect step)
    await orchestrator._connect(session_id)
    results = []
    for ep, seeds in enumerate([[17, 23], [42, 55]], start=1):
        result = await orchestrator.run_episode_async(seeds=seeds)
        results.append(result)
    await orchestrator.connection.close()

    assert len(results) == 2
    assert results[0].total_reward == pytest.approx(0.5)
    assert results[1].total_reward == pytest.approx(1.0)
    # Both episodes logged to same session file
    log_path = tmp_path / f"{session_id}.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    episode_starts = [e for e in events if e["event_type"] == "episode_start"]
    assert len(episode_starts) == 2


@pytest.mark.asyncio
async def test_prior_conjectures_appear_in_system_prompt(mock_websocket_server, tmp_path: Path) -> None:
    session_id = "prior-conj-session"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={"budget_per_episode": 5, "synthesis_cadence": 8, "family_display_name": "Decimal"},
    )
    prior = [{"id": "c1", "text": "Digit mass grows with value.", "state": "open", "survival": 5}]

    async def handler(connection: ServerConnection, _: str) -> None:
        await _recv_episode_start(connection)
        await connection.send(
            _episode_ready(session_id, seeds=[1, 2], budget=5, prior_conjectures=prior)
        )
        episode_end = json.loads(await connection.recv())
        assert episode_end["type"] == "episode_end"
        await connection.send(
            json.dumps(
                {
                    "type": "episode_complete",
                    "total_reward": 0.0,
                    "reward_breakdown": {},
                    "steps": 0,
                    "syntheses": 0,
                    "conjectures_produced": 0,
                    "conjectures_board_eligible": 0,
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, handler)

    backend = ScriptedBackend(["No more conjectures to explore."])
    orchestrator = Orchestrator(
        backend=backend,
        service_url=mock_websocket_server.service_url,
        api_key=mock_websocket_server.api_key,
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    await orchestrator.run_episode_async(session_id, seeds=[1, 2])

    system_msg = backend.calls[0][0]["content"]
    assert "Digit mass grows with value." in system_msg
    assert "Expand, falsify, or refine" in system_msg


def test_parse_model_response_handles_stop_fences_and_text_tool_calls(tmp_path: Path) -> None:
    backend = ScriptedBackend([])
    orchestrator = Orchestrator(
        backend=backend,
        service_url="ws://localhost:9090",
        api_key="test-key",
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    parsed_stop = orchestrator._parse_model_response("  No more conjectures to explore.  ")
    parsed_json = orchestrator._parse_model_response(
        "```json\n{\"tool\": \"mc.encode\", \"args\": {\"value\": 21}}\n```"
    )
    parsed_text = orchestrator._parse_model_response('[TOOL CALL] mc.compare {"lhs": "h1", "rhs": "h2"}')
    parsed_synthesis = orchestrator._parse_model_response("Conjecture: carry mass grows with depth.")

    assert isinstance(parsed_stop, EpisodeEnd)
    assert isinstance(parsed_json, ToolCall)
    assert parsed_json.tool == "mc.encode"
    assert parsed_json.args == {"value": 21}
    assert isinstance(parsed_text, ToolCall)
    assert parsed_text.tool == "mc.compare"
    assert parsed_text.args == {"lhs": "h1", "rhs": "h2"}
    assert isinstance(parsed_synthesis, Synthesis)


def test_parse_model_response_accepts_batchable_tool_batches(tmp_path: Path) -> None:
    backend = ScriptedBackend([])
    orchestrator = Orchestrator(
        backend=backend,
        service_url="ws://localhost:9090",
        api_key="test-key",
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    parsed_batch = orchestrator._parse_model_response(
        _batch_response(
            {"tool": "mc.encode", "args": {"value": 17}},
            {"tool": "mc.capabilities", "args": {}},
            {"tool": "mc.session_open", "args": {"label": "session-a"}},
        )
    )

    assert isinstance(parsed_batch, _ToolBatch)
    assert [tool_call.tool for tool_call in parsed_batch.tool_calls] == [
        "mc.encode",
        "mc.capabilities",
        "mc.session_open",
    ]
    assert parsed_batch.tool_calls[0].args == {"value": 17}
    assert parsed_batch.tool_calls[1].args == {}
    assert parsed_batch.tool_calls[2].args == {"label": "session-a"}


def test_parse_model_response_rejects_mixed_tool_batches(tmp_path: Path) -> None:
    backend = ScriptedBackend([])
    orchestrator = Orchestrator(
        backend=backend,
        service_url="ws://localhost:9090",
        api_key="test-key",
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    parsed_batch = orchestrator._parse_model_response(
        _batch_response(
            {"tool": "mc.encode", "args": {"value": 17}},
            {"tool": "mc.compare", "args": {"lhs": "h_left", "rhs": "h_right"}},
        )
    )

    assert isinstance(parsed_batch, Synthesis)
    assert "mc.compare" in parsed_batch.text


def test_system_prompt_excludes_config_field(tmp_path: Path) -> None:
    """The system prompt must never instruct the model to include a 'config' field."""
    backend = ScriptedBackend([])
    orchestrator = Orchestrator(
        backend=backend,
        service_url="ws://localhost:9090",
        api_key="test-key",
        config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )
    orchestrator.family_display_name = "Zeckendorf"
    orchestrator.family_capabilities = {"arithmetic": True, "zoom": False}
    orchestrator.budget_per_episode = 40
    orchestrator.server_synthesis_cadence = 8

    prompt = orchestrator._build_system_prompt()
    # Must NOT suggest including config
    assert '"config"' not in prompt
    # Must mention the family name
    assert "Zeckendorf" in prompt
    # Must have the no-config instruction
    assert "do NOT include a 'config' field" in prompt
    assert "Prefer exactly one tool call per turn." in prompt
    assert "Default format: output exactly one JSON object and nothing else" in prompt
    assert "Same-turn batches are allowed only when every tool is one of" in prompt
    assert "`mc.session_open`" in prompt
    assert "`mc.compare` and `mc.arithmetic` operate on handles" in prompt


def test_orchestrator_defaults_to_hosted_service_without_manual_service_config(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        backend=ScriptedBackend([]),
        api_key="hosted-key",
        session_config={"enabled_tiers": ["E0", "E1", "E2"]},
        episode_config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    assert orchestrator._service_config.service_url == DEFAULT_SERVICE_URL
    assert orchestrator._service_config.session_create_url == "https://api.mc-mcp.com/v1/sessions"
    assert orchestrator.connection.service_url == DEFAULT_SERVICE_URL
    assert orchestrator.connection.api_key == "hosted-key"
    assert orchestrator.session_config.enabled_tiers == ["E0", "E1", "E2"]


def test_run_episode_sync_is_beginner_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        backend=ScriptedBackend([]),
        episode_config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )
    sentinel = EpisodeComplete(total_reward=1.0)
    captured: dict[str, object] = {}

    def _fake_run_sync(coro_factory):
        coro = coro_factory()
        captured["is_coro"] = inspect.iscoroutine(coro)
        coro.close()
        return sentinel

    monkeypatch.setattr(orchestrator, "_run_sync", _fake_run_sync)

    result = orchestrator.run_episode(seeds=[17, 23, 42])

    assert result is sentinel
    assert captured["is_coro"] is True


@pytest.mark.asyncio
async def test_run_episode_sync_rejects_active_event_loop(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        backend=ScriptedBackend([]),
        episode_config=EpisodeConfig(local_log_dir=str(tmp_path)),
    )

    with pytest.raises(RuntimeError, match="run_episode_async"):
        orchestrator.run_episode(seeds=[17, 23, 42])


def test_build_session_create_payload_uses_public_session_contract(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        backend=ScriptedBackend([]),
        session_config=SessionConfig(
            session_id="sess_123",
            enabled_tiers=["E0", "E3"],
            budget=12,
            synthesis_cadence=5,
        ),
        episode_config=EpisodeConfig(local_log_dir=str(tmp_path), max_steps=99, synthesis_cadence=77),
    )

    assert orchestrator._build_session_create_payload() == {
        "session_id": "sess_123",
        "enabled_tiers": ["E0", "E3"],
        "budget": 12,
        "synthesis_cadence": 5,
    }


def test_create_session_sync_uses_session_config_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def read(self) -> bytes:
            return b'{"session_id": "sess_123"}'

    def _fake_urlopen(req, timeout: int = 10):
        del timeout
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("mc_mcp_client.orchestrator.request.urlopen", _fake_urlopen)

    orchestrator = Orchestrator(
        backend=ScriptedBackend([]),
        api_key="svc-key",
        service_url="ws://example.test:9090",
        episode_config=EpisodeConfig(local_log_dir=str(tmp_path), max_steps=99, synthesis_cadence=77),
        session_config=SessionConfig(
            enabled_tiers=["E0", "E1", "E2"],
            budget=12,
            synthesis_cadence=5,
        ),
    )

    session_id = orchestrator._create_session_sync()

    assert session_id == "sess_123"
    assert captured["url"] == "http://example.test:9090/v1/sessions"
    assert captured["body"] == {
        "budget": 12,
        "synthesis_cadence": 5,
        "enabled_tiers": ["E0", "E1", "E2"],
    }


def test_create_session_sync_omits_family_config_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def read(self) -> bytes:
            return b'{"session_id": "sess_123"}'

    def _fake_urlopen(req, timeout: int = 10):
        del timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    monkeypatch.setattr("mc_mcp_client.orchestrator.request.urlopen", _fake_urlopen)

    orchestrator = Orchestrator(
        backend=ScriptedBackend([]),
        api_key="svc-key",
        service_url="ws://example.test:9090",
        episode_config=EpisodeConfig(local_log_dir=str(tmp_path)),
        session_config=SessionConfig(),
    )

    orchestrator._create_session_sync()

    assert "family_config" not in captured["body"]


def test_orchestrator_rejects_duplicate_episode_config_aliases(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either episode_config or config"):
        Orchestrator(
            backend=ScriptedBackend([]),
            episode_config=EpisodeConfig(local_log_dir=str(tmp_path)),
            config=EpisodeConfig(local_log_dir=str(tmp_path)),
        )
