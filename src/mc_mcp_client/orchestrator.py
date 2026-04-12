"""Thin async episode orchestrator."""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mc_mcp_client.backends.base import LLMBackend
from mc_mcp_client.config import EpisodeConfig
from mc_mcp_client.connection import Connection
from mc_mcp_client.protocol import (
    EpisodeComplete,
    EpisodeEnd,
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


class Orchestrator:
    """Runs one episode against the MC-MCP service."""

    def __init__(
        self,
        backend: LLMBackend,
        service_url: str,
        api_key: str,
        config: EpisodeConfig | None = None,
    ) -> None:
        self.backend = backend
        self.connection = Connection(service_url, api_key)
        self.config = config or EpisodeConfig()
        self._tool_counter = 0
        self._synthesis_counter = 0
        self._pending_messages: deque[Any] = deque()

    async def run_episode(
        self,
        session_id: str,
        seeds: list[int] | None = None,
    ) -> EpisodeComplete:
        """Run a single episode and return the server's completion payload."""
        self._tool_counter = 0
        self._synthesis_counter = 0
        self._pending_messages.clear()

        resolved_seeds = list(seeds) if seeds is not None else list(self.config.seeds or [])
        log_path = self._resolve_log_path(session_id)
        ready = await self._connect_session(session_id)
        messages = [
            {"role": "system", "content": self._build_system_prompt(ready)},
            {"role": "user", "content": self._build_seed_prompt(resolved_seeds)},
        ]
        current_step = 0
        pending_synthesis_prompt: str | None = None

        self._append_log_event(
            log_path,
            {
                "event_type": "episode_start",
                "step": 0,
                "tool_step": 0,
                "budget": ready.budget,
                "synthesis_cadence": ready.synthesis_cadence,
                "enabled_tiers": list(ready.enabled_tiers),
                "curriculum_stage": self.config.stage,
                "seed_integers": list(resolved_seeds),
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

                if not isinstance(parsed, ToolCall):
                    logger.warning("Model response did not parse as a tool call; requesting one retry.")
                    messages.append({"role": "assistant", "content": response_text.strip() or "[empty assistant response]"})
                    messages.append({"role": "user", "content": "Please respond with a valid tool call in JSON format."})

                    retry_text = await self.backend.generate(
                        messages,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                    retry_parsed = self._parse_model_response(retry_text)
                    messages.append({"role": "assistant", "content": retry_text.strip() or "[empty assistant response]"})

                    if isinstance(retry_parsed, EpisodeEnd):
                        return await self._finalize_episode(
                            reason=retry_parsed.reason,
                            log_path=log_path,
                        )
                    if not isinstance(retry_parsed, ToolCall):
                        logger.warning("Model failed to produce a valid tool call after retry; ending episode.")
                        return await self._finalize_episode(
                            reason="parse_error",
                            log_path=log_path,
                        )

                    parsed = retry_parsed
                    response_text = retry_text
                else:
                    messages.append({"role": "assistant", "content": response_text.strip() or self._tool_call_text(parsed)})

                parsed.id = parsed.id or self._next_tool_id()
                await self.connection.send(asdict(parsed))
                result = await self._wait_for_tool_result()
                current_step = max(current_step, result.step)

                observation = self._compress_to_observation_card(parsed.tool, result)
                messages.append({"role": "user", "content": observation})
                self._append_log_event(
                    log_path,
                    {
                        "event_type": "tool_step",
                        "step": result.step,
                        "tool_step": result.step,
                        "tool": parsed.tool,
                        "tool_name": parsed.tool,
                        "args": dict(parsed.args),
                        "arguments": dict(parsed.args),
                        "ok": result.ok,
                        "full_response": asdict(result),
                        "observation_card": observation,
                        "reward_so_far": result.reward_so_far,
                        "reward_multiplier": result.reward_multiplier,
                        "budget_remaining": result.budget_remaining,
                    },
                )

                if self._should_stop(result):
                    return await self._finalize_episode(
                        reason="budget_exhausted",
                        log_path=log_path,
                    )

                followup_complete, followup_prompt = await self._drain_followup_messages(messages)
                if followup_complete is not None:
                    self._append_episode_end_event(
                        log_path=log_path,
                        reason="server_complete",
                        complete=followup_complete,
                    )
                    return followup_complete
                pending_synthesis_prompt = followup_prompt
        finally:
            await self.connection.close()

    def _build_system_prompt(self, session_ready: SessionReady) -> str:
        """Build the thin-client system prompt."""
        enabled_tiers = self.config.enabled_tiers or session_ready.enabled_tiers
        tiers_text = ", ".join(enabled_tiers) if enabled_tiers else "server default tiers"
        stop_text = " | ".join(self.config.stop_phrases)
        return (
            "You are running a mathematical discovery episode against the MC-MCP service.\n"
            f"Curriculum stage: {self.config.stage}\n"
            f"Available tiers: {tiers_text}\n"
            "Use the available mc.* tools to probe patterns before making claims.\n"
            f"Synthesis cadence: every {session_ready.synthesis_cadence} tool calls.\n"
            f"Tool budget: {session_ready.budget} total tool calls.\n"
            "Good conjectures generalize beyond the examples seen, are falsifiable, and point to the next discriminating test.\n"
            'On normal turns respond with JSON only, for example {"tool": "mc.encode", "args": {"value": 42}}.\n'
            f"If you want to stop, say one of: {stop_text}"
        )

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

    def _parse_model_response(self, response: str) -> ToolCall | Synthesis | EpisodeEnd:
        """Parse model text into a tool call, synthesis, or episode end."""
        stripped = response.strip()
        if self._signals_stop(stripped):
            return EpisodeEnd(reason="no_more_conjectures")

        text_tool_call = self._parse_text_tool_call(stripped)
        if text_tool_call is not None:
            return text_tool_call

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
        """Check whether the episode should stop."""
        if isinstance(result, ToolResult):
            return result.budget_remaining <= 0 or result.step >= self.config.max_steps
        return False

    async def _connect_session(self, session_id: str) -> SessionReady:
        raw = await self.connection.connect(session_id)
        message = parse_server_message(raw)
        if not isinstance(message, SessionReady):
            raise TypeError(f"Expected session_ready, received {message.type!r}")
        return message

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

    def _resolve_log_path(self, session_id: str) -> Path:
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
            "When a synthesis is requested, state one concise conjecture or falsification plan."
        )

    def _build_synthesis_prompt(self, required: SynthesisRequired) -> str:
        return (
            f"[SYNTHESIS REQUIRED - step {required.step}]\n"
            f"reward_so_far={required.reward_so_far} budget_remaining={required.budget_remaining}\n"
            "State one falsifiable conjecture or a concrete falsification plan based on the observations so far."
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

    def _tool_call_text(self, tool_call: ToolCall) -> str:
        return json.dumps({"tool": tool_call.tool, "args": tool_call.args}, ensure_ascii=False)

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
