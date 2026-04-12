"""Protocol message types and serialization (CLIENT-03)."""
import json
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Client → Server ──────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    type: str = "tool_call"
    id: str = ""            # client-assigned correlation ID
    tool: str = ""          # e.g., "mc.encode"
    args: dict = field(default_factory=dict)


@dataclass
class Synthesis:
    type: str = "synthesis"
    id: str = ""
    text: str = ""


@dataclass
class EpisodeEnd:
    type: str = "episode_end"
    reason: str = ""        # no_more_conjectures | client_stop | budget_exhausted


@dataclass
class EpisodeStart:
    type: str = "episode_start"
    id: str = ""
    seeds: list[int] | None = None


@dataclass
class Pong:
    type: str = "pong"


# ── Server → Client ──────────────────────────────────────────────────────────


@dataclass
class SessionReady:
    type: str = "session_ready"
    session_id: str = ""
    enabled_tiers: list[str] = field(default_factory=list)
    budget_per_episode: int = 40
    synthesis_cadence: int = 8
    tool_count: int = 12
    family_config: dict = field(default_factory=dict)
    family_display_name: str = ""
    capabilities: dict = field(default_factory=dict)
    step: int = 0           # always 0 on connect; included for completeness


@dataclass
class EpisodeReady:
    type: str = "episode_ready"
    id: str = ""
    episode_id: str = ""
    episode_number: int = 0
    seeds: list[int] = field(default_factory=list)
    budget: int = 40
    prior_conjectures: list[dict] = field(default_factory=list)


@dataclass
class ToolResult:
    type: str = "tool_result"
    id: str = ""
    ok: bool = True
    data: dict = field(default_factory=dict)
    step: int = 0
    budget_remaining: int = 0
    reward_so_far: float = 0.0
    reward_multiplier: float = 1.0


@dataclass
class SynthesisRequired:
    type: str = "synthesis_required"
    step: int = 0
    budget_remaining: int = 0
    reward_so_far: float = 0.0
    reward_multiplier: float = 1.0


@dataclass
class SynthesisScored:
    type: str = "synthesis_scored"
    id: str = ""
    reward_after: float = 0.0
    reward_delta: float = 0.0
    conjectures_extracted: int = 0
    conjecture_ids: list[str] = field(default_factory=list)
    prior_relevant: list[dict] = field(default_factory=list)
    reward_multiplier: float = 1.0


@dataclass
class EpisodeComplete:
    type: str = "episode_complete"
    total_reward: float = 0.0
    reward_breakdown: dict = field(default_factory=dict)
    steps: int = 0
    syntheses: int = 0
    conjectures_produced: int = 0
    conjectures_board_eligible: int = 0


@dataclass
class ServerError:
    type: str = "error"
    id: str = ""            # empty string when server sends null
    code: str = ""
    message: str = ""


# ── Parsing ───────────────────────────────────────────────────────────────────

_SERVER_TYPE_MAP: dict[str, type] = {
    "session_ready": SessionReady,
    "episode_ready": EpisodeReady,
    "tool_result": ToolResult,
    "synthesis_required": SynthesisRequired,
    "synthesis_scored": SynthesisScored,
    "episode_complete": EpisodeComplete,
    "error": ServerError,
}

ServerMessage = (
    SessionReady
    | EpisodeReady
    | ToolResult
    | SynthesisRequired
    | SynthesisScored
    | EpisodeComplete
    | ServerError
)

ClientMessage = ToolCall | Synthesis | EpisodeEnd | EpisodeStart | Pong


def parse_server_message(raw: dict[str, Any]) -> ServerMessage:
    """Parse a raw dict from the server into the appropriate dataclass.

    Raises ValueError on unknown message type.
    """
    cls = _SERVER_TYPE_MAP.get(raw.get("type"))  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"Unknown server message type: {raw.get('type')!r}")
    fields = cls.__dataclass_fields__
    # Normalize None id → "" so callers can always treat id as str.
    kwargs = {k: v for k, v in raw.items() if k in fields}
    if "id" in kwargs and kwargs["id"] is None:
        kwargs["id"] = ""
    return cls(**kwargs)


def serialize_client_message(msg: ClientMessage) -> str:
    """Serialize a client message to a JSON string."""
    return json.dumps(asdict(msg))
