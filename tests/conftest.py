from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest_asyncio
import websockets
from websockets.protocol import State
from websockets.server import ServerConnection

SessionHandler = Callable[[ServerConnection, str], Awaitable[None]]
_SESSION_PATH_RE = re.compile(r"^/v1/sessions/([^/]+)/ws$")


@dataclass
class _SessionState:
    ready_message: dict[str, Any]
    handlers: deque[SessionHandler] = field(default_factory=deque)
    active_connection: ServerConnection | None = None


class MockWebSocketServer:
    def __init__(self, api_key: str = "test-api-key") -> None:
        self.api_key = api_key
        self.connection_attempts: dict[str, int] = defaultdict(int)
        self._sessions: dict[str, _SessionState] = {}
        self._server: Any | None = None
        self._port: int | None = None

    @property
    def service_url(self) -> str:
        if self._port is None:
            raise RuntimeError("Mock WebSocket server has not started.")
        return f"ws://127.0.0.1:{self._port}"

    async def start(self) -> None:
        self._server = await websockets.serve(self._handle_connection, "127.0.0.1", 0)
        self._port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()

    def add_session(
        self,
        session_id: str,
        *,
        ready_overrides: dict[str, Any] | None = None,
    ) -> None:
        ready_message = {
            "type": "session_ready",
            "session_id": session_id,
            "enabled_tiers": ["E0", "E1", "E2"],
            "budget": 40,
            "synthesis_cadence": 8,
            "tool_count": 12,
            "step": 0,
        }
        if ready_overrides is not None:
            ready_message.update(ready_overrides)
        self._sessions[session_id] = _SessionState(ready_message=ready_message)

    def queue_handler(self, session_id: str, handler: SessionHandler) -> None:
        self._sessions[session_id].handlers.append(handler)

    async def _handle_connection(self, connection: ServerConnection) -> None:
        session_id = self._extract_session_id(connection.request.path)
        self.connection_attempts[session_id] += 1

        auth_header = connection.request.headers.get("Authorization")
        if auth_header != f"Bearer {self.api_key}":
            await connection.close(code=4001, reason="AUTH_FAILED")
            return

        session = self._sessions.get(session_id)
        if session is None:
            await connection.close(code=4004, reason="SESSION_NOT_FOUND")
            return

        current = session.active_connection
        if current is not None and getattr(current, "state", None) == State.OPEN:
            await connection.close(code=4009, reason="SESSION_CONFLICT")
            return

        session.active_connection = connection
        try:
            await connection.send(json.dumps(session.ready_message))
            if session.handlers:
                handler = session.handlers.popleft()
                await handler(connection, session_id)
            else:
                await connection.wait_closed()
        finally:
            if session.active_connection is connection:
                session.active_connection = None

    def _extract_session_id(self, path: str) -> str:
        match = _SESSION_PATH_RE.match(path)
        if match is None:
            raise RuntimeError(f"Unexpected WebSocket path: {path}")
        return match.group(1)


@pytest_asyncio.fixture
async def mock_websocket_server() -> AsyncIterator[MockWebSocketServer]:
    server = MockWebSocketServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()
