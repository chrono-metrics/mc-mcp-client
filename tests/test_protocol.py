"""Round-trip tests for protocol message types and serialization."""
import json
from dataclasses import asdict

import pytest

from mc_mcp_client.protocol import (
    EpisodeComplete,
    EpisodeEnd,
    EpisodeReady,
    EpisodeStart,
    Pong,
    ServerError,
    SessionReady,
    SynthesisRequired,
    SynthesisScored,
    Synthesis,
    ToolCall,
    ToolResult,
    parse_server_message,
    serialize_client_message,
)


# ── Client message serialization ─────────────────────────────────────────────


def test_tool_call_serialize():
    msg = ToolCall(id="c1", tool="mc.encode", args={"value": 42})
    payload = json.loads(serialize_client_message(msg))
    assert payload == {"type": "tool_call", "id": "c1", "tool": "mc.encode", "args": {"value": 42}}


def test_synthesis_serialize():
    msg = Synthesis(id="s1", text="All Zeckendorf representations exhibit monotonic digit mass.")
    payload = json.loads(serialize_client_message(msg))
    assert payload == {
        "type": "synthesis",
        "id": "s1",
        "text": "All Zeckendorf representations exhibit monotonic digit mass.",
    }


def test_episode_end_serialize():
    for reason in ("no_more_conjectures", "client_stop", "budget_exhausted"):
        msg = EpisodeEnd(reason=reason)
        payload = json.loads(serialize_client_message(msg))
        assert payload == {"type": "episode_end", "reason": reason}


def test_episode_start_serialize():
    msg = EpisodeStart(id="ep_1", seeds=[17, 23, 42])
    payload = json.loads(serialize_client_message(msg))
    assert payload == {"type": "episode_start", "id": "ep_1", "seeds": [17, 23, 42]}


def test_episode_start_serialize_no_seeds():
    msg = EpisodeStart(id="ep_2")
    payload = json.loads(serialize_client_message(msg))
    assert payload == {"type": "episode_start", "id": "ep_2", "seeds": None}


def test_pong_serialize():
    payload = json.loads(serialize_client_message(Pong()))
    assert payload == {"type": "pong"}


# ── Server message parsing ────────────────────────────────────────────────────


def test_parse_session_ready():
    raw = {
        "type": "session_ready",
        "session_id": "abc123",
        "enabled_tiers": ["E0", "E1", "E2"],
        "budget_per_episode": 40,
        "synthesis_cadence": 8,
        "tool_count": 14,
        "family_display_name": "Zeckendorf",
        "family_config": {"ladder_mode": "standard"},
        "capabilities": {"arithmetic": True, "zoom": False},
        "step": 0,
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, SessionReady)
    assert msg.session_id == "abc123"
    assert msg.enabled_tiers == ["E0", "E1", "E2"]
    assert msg.budget_per_episode == 40
    assert msg.synthesis_cadence == 8
    assert msg.tool_count == 14
    assert msg.family_display_name == "Zeckendorf"
    assert msg.family_config == {"ladder_mode": "standard"}
    assert msg.capabilities == {"arithmetic": True, "zoom": False}
    assert msg.step == 0


def test_parse_episode_ready():
    raw = {
        "type": "episode_ready",
        "id": "ep_1",
        "episode_id": "epid_abc",
        "episode_number": 1,
        "seeds": [17, 23, 42],
        "budget": 40,
        "prior_conjectures": [{"id": "c1", "text": "Foo", "state": "open", "survival": 3}],
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, EpisodeReady)
    assert msg.id == "ep_1"
    assert msg.episode_id == "epid_abc"
    assert msg.episode_number == 1
    assert msg.seeds == [17, 23, 42]
    assert msg.budget == 40
    assert len(msg.prior_conjectures) == 1
    assert msg.prior_conjectures[0]["text"] == "Foo"


def test_episode_ready_defaults():
    msg = EpisodeReady()
    assert msg.budget == 40
    assert msg.seeds == []
    assert msg.prior_conjectures == []
    assert msg.episode_number == 0


def test_parse_tool_result():
    raw = {
        "type": "tool_result",
        "id": "c1",
        "ok": True,
        "data": {"handle": "h_abc12345", "summary": {"depth_used": 8}},
        "step": 1,
        "budget_remaining": 39,
        "reward_so_far": 0.5,
        "reward_multiplier": 1.0,
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, ToolResult)
    assert msg.id == "c1"
    assert msg.ok is True
    assert msg.data["handle"] == "h_abc12345"
    assert msg.step == 1
    assert msg.budget_remaining == 39
    assert msg.reward_so_far == 0.5
    assert msg.reward_multiplier == 1.0


def test_parse_tool_result_error():
    raw = {
        "type": "tool_result",
        "id": "c2",
        "ok": False,
        "data": {"error": "INVALID_HANDLE", "detail": "Handle may have been evicted."},
        "step": 3,
        "budget_remaining": 37,
        "reward_so_far": 0.0,
        "reward_multiplier": 1.0,
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, ToolResult)
    assert msg.ok is False
    assert msg.data["error"] == "INVALID_HANDLE"


def test_parse_synthesis_required():
    raw = {
        "type": "synthesis_required",
        "step": 8,
        "budget_remaining": 32,
        "reward_so_far": 2.3,
        "reward_multiplier": 1.0,
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, SynthesisRequired)
    assert msg.step == 8
    assert msg.budget_remaining == 32
    assert msg.reward_so_far == pytest.approx(2.3)
    assert msg.reward_multiplier == 1.0


def test_parse_synthesis_scored():
    raw = {
        "type": "synthesis_scored",
        "id": "s1",
        "reward_after": 2.8,
        "reward_delta": 0.5,
        "conjectures_extracted": 1,
        "conjecture_ids": ["conj_xyz789ab"],
        "prior_relevant": [],
        "reward_multiplier": 1.0,
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, SynthesisScored)
    assert msg.id == "s1"
    assert msg.reward_after == pytest.approx(2.8)
    assert msg.reward_delta == pytest.approx(0.5)
    assert msg.conjectures_extracted == 1
    assert msg.conjecture_ids == ["conj_xyz789ab"]
    assert msg.prior_relevant == []


def test_parse_episode_complete():
    raw = {
        "type": "episode_complete",
        "total_reward": 3.2,
        "reward_breakdown": {"reward_base": 3.2, "synthesis_skip_penalty": 0.0},
        "steps": 8,
        "syntheses": 1,
        "conjectures_produced": 1,
        "conjectures_board_eligible": 1,
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, EpisodeComplete)
    assert msg.total_reward == pytest.approx(3.2)
    assert msg.steps == 8
    assert msg.syntheses == 1
    assert msg.conjectures_produced == 1
    assert msg.conjectures_board_eligible == 1


def test_parse_server_error():
    raw = {
        "type": "error",
        "id": "c1",
        "code": "INVALID_TOOL",
        "message": "Unknown tool: mc.frobnicate",
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, ServerError)
    assert msg.id == "c1"
    assert msg.code == "INVALID_TOOL"
    assert msg.message == "Unknown tool: mc.frobnicate"


def test_parse_server_error_null_id():
    """Server may send id: null for errors not tied to a specific request."""
    raw = {"type": "error", "id": None, "code": "INVALID_MESSAGE", "message": "Bad JSON"}
    msg = parse_server_message(raw)
    assert isinstance(msg, ServerError)
    assert msg.id == ""  # normalized to empty string


def test_parse_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown server message type"):
        parse_server_message({"type": "bogus_type"})


def test_parse_missing_type_raises():
    with pytest.raises(ValueError, match="Unknown server message type"):
        parse_server_message({"tool": "mc.encode"})


# ── Round-trip: construct → serialize → parse ─────────────────────────────────


def test_tool_call_roundtrip():
    original = ToolCall(id="rt1", tool="mc.arithmetic", args={"op": "add", "a": "h_aa", "b": "h_bb"})
    raw = json.loads(serialize_client_message(original))
    # Client messages are not parsed back via parse_server_message,
    # but we verify the dict round-trips through asdict correctly.
    reconstructed = ToolCall(**raw)
    assert reconstructed == original


def test_server_message_roundtrip_tool_result():
    original = ToolResult(
        id="rt2",
        ok=True,
        data={"handle": "h_deadbeef"},
        step=5,
        budget_remaining=35,
        reward_so_far=1.5,
        reward_multiplier=0.85,
    )
    raw = asdict(original)
    parsed = parse_server_message(raw)
    assert parsed == original


def test_server_message_roundtrip_synthesis_scored():
    original = SynthesisScored(
        id="s2",
        reward_after=4.1,
        reward_delta=1.2,
        conjectures_extracted=2,
        conjecture_ids=["conj_aaa", "conj_bbb"],
        prior_relevant=[{"id": "conj_old", "similarity": 0.7}],
        reward_multiplier=1.0,
    )
    raw = asdict(original)
    parsed = parse_server_message(raw)
    assert parsed == original


def test_server_message_roundtrip_episode_complete():
    original = EpisodeComplete(
        total_reward=5.5,
        reward_breakdown={"reward_base": 5.5, "synthesis_skip_penalty": 0.0},
        steps=16,
        syntheses=2,
        conjectures_produced=3,
        conjectures_board_eligible=2,
    )
    raw = asdict(original)
    parsed = parse_server_message(raw)
    assert parsed == original


# ── Extra fields are silently ignored ────────────────────────────────────────


def test_extra_fields_ignored():
    """parse_server_message drops unknown fields from future server versions."""
    raw = {
        "type": "tool_result",
        "id": "x",
        "ok": True,
        "data": {},
        "step": 1,
        "budget_remaining": 39,
        "reward_so_far": 0.0,
        "reward_multiplier": 1.0,
        "future_field": "some_new_value",  # unknown field
    }
    msg = parse_server_message(raw)
    assert isinstance(msg, ToolResult)
    assert not hasattr(msg, "future_field")


# ── Default values ────────────────────────────────────────────────────────────


def test_session_ready_defaults():
    msg = SessionReady(session_id="x")
    assert msg.budget_per_episode == 40
    assert msg.synthesis_cadence == 8
    assert msg.tool_count == 12
    assert msg.enabled_tiers == []
    assert msg.family_display_name == ""
    assert msg.family_config == {}
    assert msg.capabilities == {}
    assert msg.step == 0


def test_tool_result_defaults():
    msg = ToolResult()
    assert msg.ok is True
    assert msg.reward_multiplier == 1.0
    assert msg.data == {}
