from __future__ import annotations

import asyncio
import json

import pytest
from websockets.server import ServerConnection

from mc_mcp_client.connection import Connection


@pytest.mark.asyncio
async def test_connect_returns_session_ready(mock_websocket_server) -> None:
    session_id = "session-ready"
    mock_websocket_server.add_session(session_id)

    connection = Connection(mock_websocket_server.service_url, mock_websocket_server.api_key)
    ready = await connection.connect(session_id)

    assert ready["type"] == "session_ready"
    assert ready["session_id"] == session_id
    assert connection.is_connected is True

    await connection.close()
    assert connection.is_connected is False


@pytest.mark.asyncio
async def test_send_and_receive_json_messages(mock_websocket_server) -> None:
    session_id = "tool-flow"
    mock_websocket_server.add_session(session_id)

    async def tool_handler(connection: ServerConnection, _: str) -> None:
        raw_message = await asyncio.wait_for(connection.recv(), timeout=1.0)
        assert json.loads(raw_message) == {
            "type": "tool_call",
            "id": "call-1",
            "tool": "mc.encode",
            "args": {"value": 42},
        }
        await connection.send(
            json.dumps(
                {
                    "type": "tool_result",
                    "id": "call-1",
                    "ok": True,
                    "data": {"handle": "h_abc123"},
                    "step": 1,
                    "budget_remaining": 39,
                    "reward_so_far": 0.5,
                    "reward_multiplier": 1.0,
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, tool_handler)

    connection = Connection(mock_websocket_server.service_url, mock_websocket_server.api_key)
    await connection.connect(session_id)
    await connection.send(
        {
            "type": "tool_call",
            "id": "call-1",
            "tool": "mc.encode",
            "args": {"value": 42},
        }
    )

    message = await connection.recv(timeout=1.0)

    assert message == {
        "type": "tool_result",
        "id": "call-1",
        "ok": True,
        "data": {"handle": "h_abc123"},
        "step": 1,
        "budget_remaining": 39,
        "reward_so_far": 0.5,
        "reward_multiplier": 1.0,
    }

    await connection.close()


@pytest.mark.asyncio
async def test_recv_auto_responds_to_ping(mock_websocket_server) -> None:
    session_id = "ping-session"
    mock_websocket_server.add_session(session_id)
    pong_received = asyncio.Event()

    async def ping_handler(connection: ServerConnection, _: str) -> None:
        await connection.send(json.dumps({"type": "ping"}))
        raw_message = await asyncio.wait_for(connection.recv(), timeout=1.0)
        assert json.loads(raw_message) == {"type": "pong"}
        pong_received.set()
        await connection.send(json.dumps({"type": "tool_result", "id": "call-2", "ok": True, "data": {}}))
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, ping_handler)

    connection = Connection(mock_websocket_server.service_url, mock_websocket_server.api_key)
    await connection.connect(session_id)

    message = await connection.recv(timeout=1.0)

    assert pong_received.is_set()
    assert message == {"type": "tool_result", "id": "call-2", "ok": True, "data": {}}

    await connection.close()


@pytest.mark.asyncio
async def test_recv_reconnects_after_unexpected_close(mock_websocket_server, monkeypatch) -> None:
    session_id = "reconnect-session"
    mock_websocket_server.add_session(session_id)

    async def first_connection_handler(connection: ServerConnection, _: str) -> None:
        await connection.close(code=1011, reason="server crash")

    async def second_connection_handler(connection: ServerConnection, _: str) -> None:
        await connection.send(
            json.dumps(
                {
                    "type": "tool_result",
                    "id": "call-3",
                    "ok": True,
                    "data": {"reconnected": True},
                }
            )
        )
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, first_connection_handler)
    mock_websocket_server.queue_handler(session_id, second_connection_handler)

    recorded_delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fast_sleep(delay: float) -> None:
        recorded_delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("mc_mcp_client.connection.asyncio.sleep", fast_sleep)

    connection = Connection(
        mock_websocket_server.service_url,
        mock_websocket_server.api_key,
        max_reconnect_attempts=3,
    )
    await connection.connect(session_id)

    message = await connection.recv(timeout=1.0)

    assert message == {
        "type": "tool_result",
        "id": "call-3",
        "ok": True,
        "data": {"reconnected": True},
    }
    assert mock_websocket_server.connection_attempts[session_id] == 2
    assert 1.0 in recorded_delays

    await connection.close()


@pytest.mark.asyncio
async def test_connect_auth_failure_raises_without_reconnect(mock_websocket_server) -> None:
    session_id = "auth-failure"
    mock_websocket_server.add_session(session_id)

    connection = Connection(mock_websocket_server.service_url, "bad-api-key")

    with pytest.raises(ConnectionError, match="Authentication failed"):
        await connection.connect(session_id)

    assert mock_websocket_server.connection_attempts[session_id] == 1
    assert connection.is_connected is False


@pytest.mark.asyncio
async def test_recv_timeout_raises_timeout_error(mock_websocket_server) -> None:
    session_id = "timeout-session"
    mock_websocket_server.add_session(session_id)

    async def idle_handler(connection: ServerConnection, _: str) -> None:
        await connection.wait_closed()

    mock_websocket_server.queue_handler(session_id, idle_handler)

    connection = Connection(mock_websocket_server.service_url, mock_websocket_server.api_key)
    await connection.connect(session_id)

    with pytest.raises(TimeoutError, match="Timed out"):
        await connection.recv(timeout=0.05)

    await connection.close()
