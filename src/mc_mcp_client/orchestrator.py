"""Thin episode orchestrator with sync and async entrypoints."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from mc_mcp_client.backends.base import LLMBackend
from mc_mcp_client.config import (
    DEFAULT_SERVICE_URL,
    EpisodeConfig,
    ServiceConfig,
    SessionConfig,
)
from mc_mcp_client.connection import Connection
from mc_mcp_client.protocol import (
    EpisodeComplete,
    EpisodeEnd,
    EpisodeReady,
    EpisodeStart,
    ServerError,
    SessionReady,
    Synthesis,
    SynthesisRequired,
    SynthesisScored,
    ToolCall,
    ToolResult,
    parse_server_message,
)

logger = logging.getLogger(__name__)

_FOLLOWUP_TIMEOUT = 0.05
_TEXT_TOOL_CALL_PATTERN = re.compile(
    r"\[?\s*(?:TOOL CALL|tool call)\s*\]?\s*((?:mc|decimal)\.[A-Za-z0-9_]+)\s*(\{.*\})",
    re.DOTALL,
)
_BATCHABLE_TOOLS = frozenset({"encode", "capabilities", "session_open", "hist_calibrate"})


@dataclass
class _ToolBatch:
    tool_calls: list[ToolCall]


class Orchestrator:
    """Runs episodes against the MC-MCP service.

    `run_episode()` and `run_session()` are synchronous convenience entrypoints
    for scripts and quickstarts. Use `run_episode_async()` and
    `run_session_async()` when integrating into an existing event loop.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        api_key: str = "",
        session_config: SessionConfig | dict[str, Any] | None = None,
        service_url: str = DEFAULT_SERVICE_URL,
        episode_config: EpisodeConfig | None = None,
        config: EpisodeConfig | None = None,
        service_config: ServiceConfig | None = None,
    ) -> None:
        if episode_config is not None and config is not None:
            raise ValueError("Pass either episode_config or config, not both.")

        resolved_service_config = service_config or ServiceConfig(
            service_url=service_url or DEFAULT_SERVICE_URL,
            api_key=api_key,
        )

        self._service_config = resolved_service_config
        self.connection = Connection(
            resolved_service_config.service_url,
            resolved_service_config.api_key,
        )

        self.backend = backend
        self.config = episode_config or config or EpisodeConfig()
        self.session_config = self._coerce_session_config(session_config)

        # Session-level state (set on connect)
        self.enabled_tiers: list[str] = list(self.session_config.enabled_tiers)
        self.family_display_name: str = ""
        self.family_capabilities: dict[str, Any] = {}
        self.family_config: dict[str, Any] = {}
        self.budget_per_episode: int = self.session_config.budget
        self.server_synthesis_cadence: int = self.session_config.synthesis_cadence
        self._session_id: str | None = None

        # Episode-level state (set on each run_episode_async)
        self._prior_conjectures: list[dict] = []
        self._current_episode_budget: int = self.config.max_steps

        # Counters / buffers
        self._tool_counter = 0
        self._synthesis_counter = 0
        self._episode_counter = 0
        self._pending_messages: deque[Any] = deque()

    def run_session(
        self,
        n_episodes: int,
        seeds_per_episode: list[list[int]] | None = None,
    ) -> list[EpisodeComplete]:
        """Run multiple episodes synchronously.

        This is the beginner-facing public entrypoint for script-style use.
        Use `await run_session_async(...)` inside existing asyncio code.
        """

        return self._run_sync(
            lambda: self.run_session_async(
                n_episodes=n_episodes,
                seeds_per_episode=seeds_per_episode,
            )
        )

    async def run_session_async(
        self,
        n_episodes: int,
        seeds_per_episode: list[list[int]] | None = None,
    ) -> list[EpisodeComplete]:
        """Connect once and run N episodes; histograms accumulate across episodes."""
        await self._connect()
        results: list[EpisodeComplete] = []
        try:
            for i in range(n_episodes):
                seeds = seeds_per_episode[i] if seeds_per_episode else None
                result = await self.run_episode_async(seeds=seeds)
                results.append(result)
                logger.info(
                    "Episode %s/%s complete: reward=%.2f conjectures=%s",
                    i + 1,
                    n_episodes,
                    result.total_reward,
                    result.conjectures_produced,
                )
        finally:
            await self.connection.close()
        return results

    def run_episode(
        self,
        session_id: str | None = None,
        seeds: list[int] | None = None,
    ) -> EpisodeComplete:
        """Run a single episode synchronously.

        This is the beginner-facing public entrypoint for script-style use.
        Use `await run_episode_async(...)` inside existing asyncio code.
        """

        return self._run_sync(
            lambda: self.run_episode_async(
                session_id=session_id,
                seeds=seeds,
            )
        )

    async def run_episode_async(
        self,
        session_id: str | None = None,
        seeds: list[int] | None = None,
    ) -> EpisodeComplete:
        """Run a single episode and return the server's completion payload.

        If the connection is not open, connects first (and closes when done).
        Pass ``session_id`` to connect to a specific session; omit to
        auto-create one via the configured public session-create endpoint.
        """
        owns_connection = not self.connection.is_connected
        if owns_connection:
            await self._connect(session_id)

        self._tool_counter = 0
        self._synthesis_counter = 0
        self._pending_messages.clear()

        log_path = self._resolve_log_path()

        # ── Episode start handshake ────────────────────────────────────────
        resolved_seeds = list(seeds) if seeds is not None else list(self.config.seeds or [])
        ep_start = EpisodeStart(id=self._next_episode_id(), seeds=resolved_seeds or None)
        await self.connection.send(asdict(ep_start))
        ep_ready = await self._wait_for_episode_ready()

        self._prior_conjectures = list(ep_ready.prior_conjectures)
        self._current_episode_budget = ep_ready.budget
        episode_seeds = ep_ready.seeds or resolved_seeds

        # ── Conversation bootstrap ─────────────────────────────────────────
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_seed_prompt(episode_seeds)},
        ]
        current_step = 0
        pending_synthesis_prompt: str | None = None

        self._append_log_event(
            log_path,
            {
                "event_type": "episode_start",
                "step": 0,
                "tool_step": 0,
                "budget": ep_ready.budget,
                "synthesis_cadence": self.server_synthesis_cadence,
                "enabled_tiers": list(self.enabled_tiers),
                "family_display_name": self.family_display_name,
                "curriculum_stage": self.config.stage,
                "seed_integers": list(episode_seeds),
                "episode_number": ep_ready.episode_number,
                "prior_conjecture_count": len(self._prior_conjectures),
            },
        )

        try:
            while True:
                if pending_synthesis_prompt is not None:
                    synthesis_text = await self.backend.generate(
                        messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                    parsed = self._parse_model_response(synthesis_text)
                    normalized_synthesis = synthesis_text.strip() or "[empty synthesis response]"
                    messages.append({"role": "assistant", "content": normalized_synthesis})

                    if isinstance(parsed, EpisodeEnd):
                        return await self._finalize_episode(
                            reason=parsed.reason,
                            log_path=log_path,
                        )

                    synthesis = Synthesis(
                        id=self._next_synthesis_id(),
                        text=normalized_synthesis,
                    )
                    await self.connection.send(asdict(synthesis))
                    scored = await self._wait_for_synthesis_scored()
                    self._append_log_event(
                        log_path,
                        {
                            "event_type": "synthesis",
                            "step": current_step,
                            "tool_step": current_step,
                            "response": normalized_synthesis,
                            "synthesis_text": normalized_synthesis,
                            "prompt": pending_synthesis_prompt,
                            "reward_multiplier": scored.reward_multiplier,
                            "server_response": asdict(scored),
                            "conjecture_ids": list(scored.conjecture_ids),
                        },
                    )
                    messages.append({"role": "user", "content": self._format_synthesis_feedback(scored)})
                    pending_synthesis_prompt = None

                    followup_complete, followup_prompt = await self._drain_followup_messages(messages)
                    if followup_complete is not None:
                        self._append_episode_end_event(
                            log_path=log_path,
                            reason="server_complete",
                            complete=followup_complete,
                        )
                        return followup_complete
                    pending_synthesis_prompt = followup_prompt
                    continue

                response_text = await self.backend.generate(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                parsed = self._parse_model_response(response_text)

                if isinstance(parsed, EpisodeEnd):
                    messages.append({"role": "assistant", "content": response_text.strip() or parsed.reason})
                    return await self._finalize_episode(
                        reason=parsed.reason,
                        log_path=log_path,
                    )

                if not isinstance(parsed, (ToolCall, _ToolBatch)):
                    logger.warning("Model response did not parse as a tool call; requesting one retry.")
                    messages.append({"role": "assistant", "content": response_text.strip() or "[empty assistant response]"})
                    messages.append({"role": "user", "content": self._build_retry_prompt()})

                    retry_text = await self.backend.generate(
                        messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                    retry_parsed = self._parse_model_response(retry_text)

                    if isinstance(retry_parsed, EpisodeEnd):
                        messages.append({"role": "assistant", "content": retry_text.strip() or retry_parsed.reason})
                        return await self._finalize_episode(
                            reason=retry_parsed.reason,
                            log_path=log_path,
                        )
                    if not isinstance(retry_parsed, (ToolCall, _ToolBatch)):
                        messages.append({"role": "assistant", "content": retry_text.strip() or "[empty assistant response]"})
                        logger.warning("Model failed to produce a valid tool call after retry; ending episode.")
                        return await self._finalize_episode(
                            reason="parse_error",
                            log_path=log_path,
                        )

                    parsed = retry_parsed
                    response_text = retry_text
                current_step, turn_complete, pending_synthesis_prompt = await self._execute_tool_turn(
                    parsed=parsed,
                    response_text=response_text,
                    messages=messages,
                    log_path=log_path,
                    current_step=current_step,
                )
                if turn_complete is not None:
                    return turn_complete
        finally:
            if owns_connection:
                await self.connection.close()

    # ── Connection management ─────────────────────────────────────────────────

    async def _connect(self, session_id: str | None = None) -> SessionReady:
        """Open the WebSocket and receive session_ready; store session-level state."""
        if session_id is None:
            session_id = await self._create_session_via_http()

        raw = await self.connection.connect(session_id)
        message = parse_server_message(raw)
        if not isinstance(message, SessionReady):
            raise TypeError(f"Expected session_ready, received {message.type!r}")

        self._session_id = session_id
        self.enabled_tiers = list(message.enabled_tiers)
        self.family_display_name = message.family_display_name
        self.family_capabilities = dict(message.capabilities)
        self.family_config = dict(message.family_config)
        self.budget_per_episode = message.budget_per_episode
        self.server_synthesis_cadence = message.synthesis_cadence

        logger.info(
            "Session %s ready: family=%r budget=%s cadence=%s",
            session_id,
            self.family_display_name or "(unset)",
            self.budget_per_episode,
            self.server_synthesis_cadence,
        )
        return message

    async def _create_session_via_http(self) -> str:
        if not self._service_config.session_create_url:
            raise RuntimeError("Cannot auto-create a session: service config is missing session_create_url.")
        return await asyncio.to_thread(self._create_session_sync)

    def _build_session_create_payload(self) -> dict[str, Any]:
        """Build the public thin-client session-create payload."""

        return self.session_config.to_create_payload()

    def _create_session_sync(self) -> str:
        cfg = self._service_config
        payload = self._build_session_create_payload()

        req = request.Request(
            cfg.session_create_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=10) as response:
                raw = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"Failed to create session at {cfg.session_create_url}: {exc}") from exc

        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or "session_id" not in parsed:
            raise RuntimeError("Session creation response did not include a session_id.")
        return str(parsed["session_id"])

    @staticmethod
    def _run_sync(coro_factory: Callable[[], Awaitable[Any]]) -> Any:
        """Run a coroutine from sync code and reject nested event-loop use."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())

        raise RuntimeError(
            "run_episode() and run_session() cannot be called from an active event loop. "
            "Use await run_episode_async(...) or await run_session_async(...) instead."
        )

    @staticmethod
    def _coerce_session_config(
        session_config: SessionConfig | dict[str, Any] | None,
    ) -> SessionConfig:
        if session_config is None:
            return SessionConfig()
        if isinstance(session_config, SessionConfig):
            return session_config
        if isinstance(session_config, dict):
            return SessionConfig(**session_config)
        raise TypeError("session_config must be a SessionConfig, mapping, or None")

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build the system prompt from session-level state."""
        family_name = self.family_display_name or "mathematical"
        caps = self.family_capabilities
        if caps:
            supported_ops = ", ".join(k for k, v in caps.items() if v)
            unsupported_ops = ", ".join(k for k, v in caps.items() if not v)
            ops_text = f"Supported operations: {supported_ops}." if supported_ops else ""
            if unsupported_ops:
                ops_text += f"\nNot supported: {unsupported_ops}."
        else:
            ops_text = "Use mc.* tools to probe the representation."

        stop_text = " | ".join(self.config.stop_phrases)
        prior_text = self._format_prior_conjectures()

        return (
            f"You are a mathematical discovery agent exploring the {family_name} representation family.\n\n"
            "Follow the active family and tools for this run. "
            "You will receive only compressed observation cards after each tool call. "
            "Handles are opaque ids: reuse them exactly and never infer meaning from the handle string.\n\n"
            f"{ops_text}\n\n"
            f"Budget: {self.budget_per_episode} tool calls per episode. "
            f"Synthesis cadence: every {self.server_synthesis_cadence} tool calls.\n\n"
            "Tool protocol:\n"
            "- The active family/config for this run is fixed. Provide only semantic arguments; do NOT include a 'config' field.\n"
            "- Emit exactly one tool call per turn.\n"
            "- Output exactly one JSON object and nothing else: no prose, no markdown fences, no backticks, no surrounding commentary.\n"
            '  Example: {"tool": "mc.encode", "args": {"n": 42}}\n'
            '  Example: {"tool": "mc.inspect", "args": {"handle": "h_abc", "view": "prefix"}}\n'
            '  Example: {"tool": "mc.arithmetic", "args": {"op": "add", "lhs": "h_abc", "rhs": "h_def"}}\n'
            "- `mc.compare` and `mc.arithmetic` operate on handles. If you want to compare or combine a fresh integer, encode it first.\n"
            "- Do not repeat the same tool call with the same args unless the last result created a new reason.\n\n"
            "When synthesis is requested, do not call a tool. Write plain text only: one falsifiable conjecture or one concrete falsification plan grounded in the observed cards.\n"
            "Good conjectures generalize beyond the examples seen, are falsifiable, and point to the next discriminating test.\n"
            f"To stop, say exactly one of: {stop_text}\n"
            f"{prior_text}"
        )

    def _build_retry_prompt(self) -> str:
        return (
            "Respond again with exactly one tool call as a single JSON object and nothing else. "
            "Do not include prose, markdown fences, or a `config` field. "
            'Format: {"tool": "mc.encode", "args": {"n": 42}}. '
            "If you truly want to stop, say exactly: I have no more conjectures to explore."
        )

    def _format_prior_conjectures(self) -> str:
        if not self._prior_conjectures:
            return ""
        lines = ["\nYour prior conjectures in this family:"]
        for c in self._prior_conjectures:
            state = c.get("state", "unknown")
            text = c.get("text", "")
            survival = c.get("survival", 0)
            lines.append(f'- [{state}] "{text}" (survived {survival} tests)')
        lines.append(
            "Expand, falsify, or refine these. Restating without extension is not valuable."
        )
        return "\n".join(lines)

    def _compress_to_observation_card(self, tool: str, result: ToolResult) -> str:
        """Compress a tool result into a 4-line observation card."""
        data = result.data if isinstance(result.data, dict) else {}
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        family = self._first_non_empty(
            data.get("family"),
            summary.get("family") if isinstance(summary, dict) else None,
            tool.split(".", 1)[0] if "." in tool else None,
            "-",
        )
        depth = self._first_non_empty(
            data.get("depth"),
            data.get("depth_used"),
            data.get("zoom_depth"),
            summary.get("depth") if isinstance(summary, dict) else None,
            summary.get("depth_used") if isinstance(summary, dict) else None,
            summary.get("zoom_depth") if isinstance(summary, dict) else None,
            "-",
        )
        handle = self._first_non_empty(
            data.get("handle"),
            data.get("session_id"),
            "-",
        )

        return "\n".join(
            [
                f"tool={tool}",
                f"family={family} depth={depth} handle={handle}",
                f"summary={self._summarize_tool_result(result)}",
                f"reward_so_far={result.reward_so_far} budget={result.budget_remaining}",
            ]
        )

    def _format_synthesis_feedback(self, scored: SynthesisScored) -> str:
        """Format synthesis scoring feedback as a user message."""
        conjecture_ids = ", ".join(scored.conjecture_ids) if scored.conjecture_ids else "-"
        prior_relevant = ", ".join(
            str(item.get("id", "?")) for item in scored.prior_relevant if isinstance(item, dict)
        ) or "none"
        return (
            "Synthesis scored.\n"
            f"reward_delta={scored.reward_delta} reward_after={scored.reward_after}\n"
            f"conjectures_extracted={scored.conjectures_extracted} conjecture_ids={conjecture_ids}\n"
            f"prior_relevant={prior_relevant} reward_multiplier={scored.reward_multiplier}"
        )

    def _parse_model_response(
        self,
        response: str,
    ) -> ToolCall | _ToolBatch | Synthesis | EpisodeEnd:
        """Parse model text into a tool call, synthesis, or episode end."""
        stripped = response.strip()
        if self._signals_stop(stripped):
            return EpisodeEnd(reason="no_more_conjectures")

        text_tool_call = self._parse_text_tool_call(stripped)
        if text_tool_call is not None:
            return text_tool_call

        streamed_tool_calls = self._parse_json_tool_calls(stripped)
        if streamed_tool_calls is not None:
            if len(streamed_tool_calls) == 1:
                return streamed_tool_calls[0]
            if all(self._is_batchable_tool(tool_call.tool) for tool_call in streamed_tool_calls):
                return _ToolBatch(tool_calls=streamed_tool_calls)
            return Synthesis(text=stripped)

        try:
            payload = self._extract_json_object(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            return Synthesis(text=stripped)

        tool_name = payload.get("tool") or payload.get("name")
        args = payload.get("args", payload.get("arguments", payload.get("input", {})))
        if isinstance(tool_name, str) and isinstance(args, dict):
            return ToolCall(
                id=str(payload.get("id") or ""),
                tool=tool_name,
                args=dict(args),
            )
        return Synthesis(text=stripped)

    def _should_stop(self, result: ToolResult | SynthesisScored) -> bool:
        if isinstance(result, ToolResult):
            return result.budget_remaining <= 0 or result.step >= self._current_episode_budget
        return False

    # ── Server message helpers ────────────────────────────────────────────────

    async def _wait_for_episode_ready(self) -> EpisodeReady:
        skipped: list[Any] = []
        try:
            while True:
                message = await self._next_server_message()
                if isinstance(message, EpisodeReady):
                    return message
                skipped.append(message)
        finally:
            self._requeue_front(skipped)

    async def _wait_for_tool_result(self) -> ToolResult:
        skipped: list[Any] = []
        try:
            while True:
                message = await self._next_server_message()
                if isinstance(message, ToolResult):
                    return message
                if isinstance(message, ServerError):
                    return ToolResult(
                        id=message.id,
                        ok=False,
                        data={"error": message.code, "detail": message.message},
                    )
                skipped.append(message)
        finally:
            self._requeue_front(skipped)

    async def _wait_for_synthesis_scored(self) -> SynthesisScored:
        skipped: list[Any] = []
        try:
            while True:
                message = await self._next_server_message()
                if isinstance(message, SynthesisScored):
                    return message
                skipped.append(message)
        finally:
            self._requeue_front(skipped)

    async def _wait_for_episode_complete(self) -> EpisodeComplete:
        skipped: list[Any] = []
        try:
            while True:
                message = await self._next_server_message()
                if isinstance(message, EpisodeComplete):
                    return message
                skipped.append(message)
        finally:
            self._requeue_front(skipped)

    async def _execute_tool_turn(
        self,
        *,
        parsed: ToolCall | _ToolBatch,
        response_text: str,
        messages: list[dict[str, str]],
        log_path: Path,
        current_step: int,
    ) -> tuple[int, EpisodeComplete | None, str | None]:
        tool_calls = [parsed] if isinstance(parsed, ToolCall) else list(parsed.tool_calls)
        assistant_content = response_text.strip() or self._tool_calls_text(tool_calls)
        messages.append({"role": "assistant", "content": assistant_content})

        observations: list[str] = []
        followup_messages: list[dict[str, str]] = []

        for tool_call in tool_calls:
            tool_call.id = tool_call.id or self._next_tool_id()
            await self.connection.send(asdict(tool_call))
            result = await self._wait_for_tool_result()
            current_step = max(current_step, result.step)

            observation = self._compress_to_observation_card(tool_call.tool, result)
            observations.append(observation)
            self._append_log_event(
                log_path,
                {
                    "event_type": "tool_step",
                    "step": result.step,
                    "tool_step": result.step,
                    "tool": tool_call.tool,
                    "tool_name": tool_call.tool,
                    "args": dict(tool_call.args),
                    "arguments": dict(tool_call.args),
                    "ok": result.ok,
                    "full_response": asdict(result),
                    "observation_card": observation,
                    "reward_so_far": result.reward_so_far,
                    "reward_multiplier": result.reward_multiplier,
                    "budget_remaining": result.budget_remaining,
                },
            )

            if self._should_stop(result):
                self._append_observations(messages, observations, followup_messages)
                complete = await self._finalize_episode(
                    reason="budget_exhausted",
                    log_path=log_path,
                )
                return current_step, complete, None

            followup_complete, followup_prompt = await self._drain_followup_messages(followup_messages)
            if followup_complete is not None:
                self._append_observations(messages, observations, followup_messages)
                self._append_episode_end_event(
                    log_path=log_path,
                    reason="server_complete",
                    complete=followup_complete,
                )
                return current_step, followup_complete, None

            if followup_prompt is not None:
                self._append_observations(messages, observations, followup_messages)
                return current_step, None, followup_prompt

        self._append_observations(messages, observations, followup_messages)
        return current_step, None, None

    async def _drain_followup_messages(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[EpisodeComplete | None, str | None]:
        synthesis_prompt: str | None = None
        while True:
            try:
                message = await self._next_server_message(timeout=_FOLLOWUP_TIMEOUT)
            except TimeoutError:
                return None, synthesis_prompt

            if isinstance(message, EpisodeComplete):
                return message, synthesis_prompt

            if isinstance(message, SynthesisRequired):
                synthesis_prompt = self._build_synthesis_prompt(message)
                messages.append({"role": "user", "content": synthesis_prompt})
                continue

            if isinstance(message, ServerError):
                logger.warning("Server protocol error: %s: %s", message.code, message.message)
                messages.append({"role": "user", "content": f"Server error: {message.code}: {message.message}"})
                continue

            self._pending_messages.appendleft(message)
            return None, synthesis_prompt

    async def _next_server_message(self, timeout: float = 120.0) -> Any:
        if self._pending_messages:
            return self._pending_messages.popleft()
        return parse_server_message(await self.connection.recv(timeout=timeout))

    async def _finalize_episode(
        self,
        *,
        reason: str,
        log_path: Path,
    ) -> EpisodeComplete:
        wire_reason = self._wire_episode_end_reason(reason)
        await self.connection.send(asdict(EpisodeEnd(reason=wire_reason)))
        complete = await self._wait_for_episode_complete()
        self._append_episode_end_event(log_path=log_path, reason=reason, complete=complete)
        return complete

    def _append_episode_end_event(
        self,
        *,
        log_path: Path,
        reason: str,
        complete: EpisodeComplete,
    ) -> None:
        self._append_log_event(
            log_path,
            {
                "event_type": "episode_end",
                "step": complete.steps,
                "tool_step": complete.steps,
                "stopped_reason": reason,
                "summary": asdict(complete),
            },
        )

    def _resolve_log_path(self) -> Path:
        session_id = self._session_id or "unknown_session"
        target_dir = Path(self.config.local_log_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{session_id}.jsonl"

    def _build_seed_prompt(self, seeds: list[int]) -> str:
        if seeds:
            seed_text = ", ".join(str(value) for value in seeds)
        else:
            seed_text = "server-selected"
        return (
            "Start the discovery episode.\n"
            f"Seed integers: {seed_text}\n"
            "You will receive compressed observation cards after each tool call.\n"
            "On tool turns, emit exactly one tool call as a single JSON object.\n"
            "When a synthesis is requested, write plain text only: one concise conjecture or falsification plan."
        )

    def _build_synthesis_prompt(self, required: SynthesisRequired) -> str:
        return (
            f"[SYNTHESIS REQUIRED - step {required.step}]\n"
            f"reward_so_far={required.reward_so_far} budget_remaining={required.budget_remaining}\n"
            "Do not call a tool or output JSON. State one falsifiable conjecture or a concrete falsification plan based on the observations so far."
        )

    def _summarize_tool_result(self, result: ToolResult) -> str:
        data = result.data if isinstance(result.data, dict) else {}
        if not result.ok:
            error_code = data.get("error", "unknown_error")
            detail = data.get("detail") or data.get("message") or ""
            if detail:
                return f"error={error_code}; detail={detail}"
            return f"error={error_code}"

        summary = data.get("summary")
        if isinstance(summary, dict) and summary:
            return "; ".join(
                f"{key}={self._stringify_scalar(value)}"
                for key, value in list(summary.items())[:3]
            )

        parts: list[str] = []
        for key, value in data.items():
            if key in {"summary", "handle", "family", "depth", "depth_used", "zoom_depth"}:
                continue
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}={self._stringify_scalar(value)}")
            if len(parts) == 3:
                break
        if parts:
            return "; ".join(parts)
        return "ok"

    def _wire_episode_end_reason(self, reason: str) -> str:
        if reason == "no_more_conjectures":
            return "no_more_conjectures"
        if reason == "budget_exhausted":
            return "budget_exhausted"
        return "client_stop"

    def _signals_stop(self, text: str) -> bool:
        normalized = " ".join(text.lower().split())
        phrases = {phrase.lower() for phrase in self.config.stop_phrases}
        phrases.add("no more conjectures to explore")
        return any(phrase in normalized for phrase in phrases)

    def _parse_text_tool_call(self, text: str) -> ToolCall | None:
        match = _TEXT_TOOL_CALL_PATTERN.search(text)
        if match is None:
            return None
        arguments = json.loads(match.group(2))
        if not isinstance(arguments, dict):
            raise TypeError("text tool call arguments must decode to an object")
        return ToolCall(tool=match.group(1), args=dict(arguments))

    def _parse_json_tool_calls(self, text: str) -> list[ToolCall] | None:
        candidate = self._unwrap_json_fence(text)
        if not candidate:
            return None

        decoder = json.JSONDecoder()
        tool_calls: list[ToolCall] = []
        index = 0

        try:
            while index < len(candidate):
                while index < len(candidate) and candidate[index].isspace():
                    index += 1
                if index >= len(candidate):
                    break

                payload, index = decoder.raw_decode(candidate, idx=index)
                if not isinstance(payload, dict):
                    return None

                tool_name = payload.get("tool") or payload.get("name")
                args = payload.get("args", payload.get("arguments", payload.get("input", {})))
                if not isinstance(tool_name, str) or not isinstance(args, dict):
                    return None

                tool_calls.append(
                    ToolCall(
                        id=str(payload.get("id") or ""),
                        tool=tool_name,
                        args=dict(args),
                    )
                )
        except json.JSONDecodeError:
            return None

        return tool_calls or None

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        if not text:
            raise ValueError("empty JSON response")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no JSON object found in response")

        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise TypeError("parsed JSON payload is not an object")
        return parsed

    def _next_tool_id(self) -> str:
        self._tool_counter += 1
        return f"call_{self._tool_counter}"

    def _next_synthesis_id(self) -> str:
        self._synthesis_counter += 1
        return f"synth_{self._synthesis_counter}"

    def _next_episode_id(self) -> str:
        self._episode_counter += 1
        return f"ep_{self._episode_counter}"

    def _tool_call_text(self, tool_call: ToolCall) -> str:
        return json.dumps({"tool": tool_call.tool, "args": tool_call.args}, ensure_ascii=False)

    def _tool_calls_text(self, tool_calls: list[ToolCall]) -> str:
        return "\n".join(self._tool_call_text(tool_call) for tool_call in tool_calls)

    def _append_log_event(self, log_path: Path, event: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True))
            handle.write("\n")

    def _requeue_front(self, messages: list[Any]) -> None:
        for message in reversed(messages):
            self._pending_messages.appendleft(message)

    @staticmethod
    def _first_non_empty(*values: Any) -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return str(value)
        return "-"

    @staticmethod
    def _stringify_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _append_observations(
        messages: list[dict[str, str]],
        observations: list[str],
        followup_messages: list[dict[str, str]],
    ) -> None:
        if observations:
            messages.append({"role": "user", "content": "\n\n".join(observations)})
        messages.extend(followup_messages)

    @staticmethod
    def _unwrap_json_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```") or not stripped.endswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) < 2 or lines[-1].strip() != "```":
            return stripped
        return "\n".join(lines[1:-1]).strip()

    @staticmethod
    def _is_batchable_tool(tool_name: str) -> bool:
        _, _, operation = tool_name.partition(".")
        return operation in _BATCHABLE_TOOLS
