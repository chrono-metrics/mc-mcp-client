from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest
from websockets.server import ServerConnection

from mc_mcp_client.backends.base import LLMBackend
from mc_mcp_client.config import EpisodeConfig
from mc_mcp_client.orchestrator import Orchestrator
from mc_mcp_client.protocol import EpisodeEnd, Synthesis, ToolCall


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


@pytest.mark.asyncio
async def test_run_episode_handles_tool_synthesis_and_stop(mock_websocket_server, tmp_path: Path) -> None:
    session_id = "episode-happy-path"
    mock_websocket_server.add_session(
        session_id,
        ready_overrides={"budget": 2, "synthesis_cadence": 1, "enabled_tiers": ["E0", "E1", "E2"]},
    )
    received_messages: list[dict] = []

    async def episode_handler(connection: ServerConnection, _: str) -> None:
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

    result = await orchestrator.run_episode(session_id, seeds=[5, 8, 13])

    assert result.total_reward == pytest.approx(1.25)
    assert result.steps == 1
    assert received_messages[0]["type"] == "tool_call"
    assert received_messages[0]["tool"] == "mc.encode"
    assert received_messages[1]["type"] == "synthesis"
    assert "Conjecture:" in received_messages[1]["text"]
    assert received_messages[2] == {"type": "episode_end", "reason": "no_more_conjectures"}

    first_prompt = backend.calls[0]
    assert first_prompt[0]["role"] == "system"
    assert "Available tiers: E0, E1, E2" in first_prompt[0]["content"]
    assert "Seed integers: 5, 8, 13" in first_prompt[1]["content"]

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
        ready_overrides={"budget": 3, "synthesis_cadence": 8},
    )
    received_messages: list[dict] = []

    async def parse_retry_handler(connection: ServerConnection, _: str) -> None:
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

    result = await orchestrator.run_episode(session_id)

    assert result.steps == 1
    assert received_messages[0]["tool"] == "mc.inspect"
    assert received_messages[0]["args"] == {"handle": "h_dead", "view": "prefix"}
    assert received_messages[1] == {"type": "episode_end", "reason": "no_more_conjectures"}
    assert any(
        message["role"] == "user" and "valid tool call in JSON format" in message["content"]
        for message in backend.calls[1]
    )


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
