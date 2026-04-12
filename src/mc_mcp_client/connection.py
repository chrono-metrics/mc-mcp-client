"""WebSocket session management."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from websockets.protocol import State

logger = logging.getLogger(__name__)

_AUTH_FAILED = 4001
_SESSION_NOT_FOUND = 4004
_SESSION_CONFLICT = 4009
_PERMANENT_CONNECT_CODES = {_AUTH_FAILED, _SESSION_NOT_FOUND, _SESSION_CONFLICT}
_PERMANENT_RECONNECT_CODES = {_AUTH_FAILED, _SESSION_NOT_FOUND}


@dataclass(slots=True)
class _ConnectFailure(ConnectionError):
    """Internal exception that preserves reconnect permanence."""

    permanent: bool
    message: str

    def __str__(self) -> str:
        return self.message


class Connection:
    """Manages a WebSocket connection to the MC-MCP service."""

    def __init__(
        self,
        service_url: str,
        api_key: str,
        ping_interval: float = 30.0,
        connect_timeout: float = 10.0,
        max_reconnect_attempts: int = 3,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.api_key = api_key
        self.ping_interval = ping_interval
        self.connect_timeout = connect_timeout
        self.max_reconnect_attempts = max_reconnect_attempts

        self._websocket: Any | None = None
        self._session_id: str | None = None
        self._closed_by_user = False
        self._state_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        """Return whether the underlying socket is currently open."""
        websocket = self._websocket
        return websocket is not None and getattr(websocket, "state", None) == State.OPEN

    async def connect(self, session_id: str) -> dict[str, Any]:
        """Connect to a session WebSocket and return the session_ready payload."""
        async with self._state_lock:
            if self.is_connected:
                raise ConnectionError("Connection is already established.")

            self._closed_by_user = False
            self._session_id = session_id

            logger.info("Connecting to session %s", session_id)
            try:
                websocket, ready = await self._open_and_handshake(
                    session_id=session_id,
                    permanent_codes=_PERMANENT_CONNECT_CODES,
                )
            except _ConnectFailure as exc:
                self._websocket = None
                self._session_id = None
                raise ConnectionError(str(exc)) from exc

            self._websocket = websocket
            logger.info("Connected to session %s", session_id)
            return ready

    async def send(self, message: dict[str, Any]) -> None:
        """Send a JSON message."""
        payload = json.dumps(message)

        for _ in range(2):
            websocket = self._websocket
            if websocket is None:
                raise ConnectionError("Connection is not established.")

            try:
                await websocket.send(payload)
                logger.debug("Sent message: %s", payload)
                return
            except ConnectionClosed as exc:
                await self._recover_from_disconnect(websocket, exc)
                continue
            except WebSocketException as exc:
                raise ConnectionError(f"Failed to send WebSocket message: {exc}") from exc

        raise ConnectionError("Connection dropped while sending the message.")

    async def recv(self, timeout: float = 120.0) -> dict[str, Any]:
        """Receive the next non-ping JSON message."""
        deadline = asyncio.get_running_loop().time() + timeout

        while True:
            websocket = self._websocket
            if websocket is None:
                raise ConnectionError("Connection is not established.")

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for a message.")

            try:
                return await self._recv_json_message(websocket, timeout=remaining)
            except ConnectionClosed as exc:
                await self._recover_from_disconnect(websocket, exc)
                continue

    async def close(self) -> None:
        """Gracefully disconnect the WebSocket."""
        async with self._state_lock:
            self._closed_by_user = True
            websocket = self._websocket
            session_id = self._session_id
            self._websocket = None
            self._session_id = None

        if websocket is not None:
            await websocket.close()

        if session_id is not None:
            logger.info("Disconnected from session %s", session_id)

    async def _recover_from_disconnect(self, websocket: Any, exc: ConnectionClosed) -> None:
        if self._closed_by_user:
            raise ConnectionError("Connection was closed.") from exc

        session_id = self._session_id
        if session_id is None:
            raise ConnectionError("Connection is not established.") from exc

        async with self._state_lock:
            if self._closed_by_user:
                raise ConnectionError("Connection was closed.") from exc

            current = self._websocket
            if current is not None and current is not websocket and getattr(current, "state", None) == State.OPEN:
                return

            self._websocket = None
            close_code, close_reason = self._close_details(exc)
            if close_code in _PERMANENT_RECONNECT_CODES:
                raise ConnectionError(self._format_close_error(close_code, close_reason)) from exc

            delay = 1.0
            last_error: Exception = ConnectionError(self._format_close_error(close_code, close_reason))

            for attempt in range(1, self.max_reconnect_attempts + 1):
                logger.info(
                    "Reconnecting to session %s (attempt %s/%s)",
                    session_id,
                    attempt,
                    self.max_reconnect_attempts,
                )
                await asyncio.sleep(delay)

                try:
                    new_websocket, _ = await self._open_and_handshake(
                        session_id=session_id,
                        permanent_codes=_PERMANENT_RECONNECT_CODES,
                    )
                except _ConnectFailure as reconnect_error:
                    last_error = reconnect_error
                    if reconnect_error.permanent:
                        raise ConnectionError(str(reconnect_error)) from reconnect_error
                    delay *= 2
                    continue
                except Exception as reconnect_error:  # pragma: no cover - defensive fallback
                    last_error = reconnect_error
                    delay *= 2
                    continue

                self._websocket = new_websocket
                logger.warning(
                    "Reconnected to session %s on attempt %s",
                    session_id,
                    attempt,
                )
                return

        raise ConnectionError(f"Failed to reconnect to session {session_id}: {last_error}") from last_error

    async def _open_and_handshake(
        self,
        session_id: str,
        permanent_codes: set[int],
    ) -> tuple[Any, dict[str, Any]]:
        try:
            websocket = await websockets.connect(
                self._session_url(session_id),
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
                open_timeout=self.connect_timeout,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_interval,
            )
        except Exception as exc:
            raise _ConnectFailure(False, f"Failed to open WebSocket connection: {exc}") from exc

        try:
            ready = await self._recv_json_message(websocket, timeout=self.connect_timeout)
        except TimeoutError as exc:
            await websocket.close()
            raise _ConnectFailure(False, "Timed out waiting for session_ready.") from exc
        except ConnectionClosed as exc:
            await websocket.close()
            close_code, close_reason = self._close_details(exc)
            permanent = close_code in permanent_codes
            raise _ConnectFailure(permanent, self._format_close_error(close_code, close_reason)) from exc
        except Exception:
            await websocket.close()
            raise

        if ready.get("type") != "session_ready":
            await websocket.close()
            raise _ConnectFailure(
                False,
                f"Expected session_ready after connecting, received {ready.get('type')!r}.",
            )

        return websocket, ready

    async def _recv_json_message(self, websocket: Any, timeout: float) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for a message.")

            try:
                raw_message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise TimeoutError("Timed out waiting for a message.") from exc

            message = self._decode_message(raw_message)
            if message.get("type") == "ping":
                await self._send_json_message(websocket, {"type": "pong"})
                continue

            return message

    async def _send_json_message(self, websocket: Any, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        await websocket.send(payload)
        logger.debug("Sent message: %s", payload)

    def _decode_message(self, raw_message: str | bytes) -> dict[str, Any]:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")

        logger.debug("Received message: %s", raw_message)

        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            raise ConnectionError("Received invalid JSON from the server.") from exc

        if not isinstance(message, dict):
            raise ConnectionError("Expected a JSON object from the server.")

        return message

    def _format_close_error(self, code: int | None, reason: str | None) -> str:
        if code == _AUTH_FAILED:
            return "Authentication failed while connecting to the session."
        if code == _SESSION_NOT_FOUND:
            return "Session not found."
        if code == _SESSION_CONFLICT:
            return "Session is already connected."
        if code is None:
            return "WebSocket connection closed unexpectedly."
        if reason:
            return f"WebSocket connection closed unexpectedly ({code}: {reason})."
        return f"WebSocket connection closed unexpectedly ({code})."

    def _close_details(self, exc: ConnectionClosed) -> tuple[int | None, str | None]:
        for close in (getattr(exc, "rcvd", None), getattr(exc, "sent", None)):
            if close is not None:
                return close.code, close.reason
        return None, None

    def _session_url(self, session_id: str) -> str:
        return f"{self.service_url}/v1/sessions/{session_id}/ws"


MCMCPConnection = Connection
