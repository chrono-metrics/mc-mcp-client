# MC-MCP WebSocket Protocol

This document is the human-readable reference for the MC-MCP gym service
protocol. The authoritative Python types live in
`src/mc_mcp_client/protocol.py`.

---

## Overview

The protocol has two layers:

1. **HTTP** — create a session and receive a WebSocket URL.
2. **WebSocket** — stream tool calls and syntheses; receive results and reward
   signals.

All WebSocket messages are UTF-8 JSON objects with a `"type"` discriminator
field.

---

## 1. Connection Establishment

### 1.1 Create a session

```
POST /v1/sessions
Authorization: Bearer <api_key>
Content-Type: application/json
```

Request body (all fields optional):

```json
{
  "session_id": "my-run-001",
  "budget": 40,
  "synthesis_cadence": 8,
  "enabled_tiers": ["E0", "E1", "E2"]
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `session_id` | auto-generated | Custom ID; omit to let the server generate one |
| `budget` | `40` | Maximum tool calls for the episode |
| `synthesis_cadence` | `8` | Tool steps between required syntheses |
| `enabled_tiers` | `["E0","E1","E2"]` | Representation tiers to enable |

Default API key for local development: `mcmcp_dev`.

Response `200 OK`:

```json
{
  "session_id": "abc123def456",
  "ws_url": "ws://localhost:9090/v1/sessions/abc123def456/ws",
  "enabled_tiers": ["E0", "E1", "E2"],
  "budget": 40,
  "synthesis_cadence": 8
}
```

### 1.2 Connect the WebSocket

Open a WebSocket to the `ws_url` returned above, sending the same bearer token:

```
GET /v1/sessions/{session_id}/ws
Authorization: Bearer <api_key>
Upgrade: websocket
```

On success the server immediately sends a `session_ready` message (see §2.1).

#### WebSocket close codes

| Code | Name | Meaning |
|------|------|---------|
| `4001` | `AUTH_FAILED` | Missing or invalid bearer token |
| `4004` | `SESSION_NOT_FOUND` | Session ID does not exist |
| `4009` | `SESSION_CONFLICT` | Another WebSocket is already connected to this session |
| `4400` | `PROTOCOL_ERROR` | Client missed two consecutive pong responses |

---

## 2. Server → Client Messages

### 2.1 `session_ready`

Sent once, immediately after the WebSocket handshake succeeds.

```json
{
  "type": "session_ready",
  "session_id": "abc123def456",
  "enabled_tiers": ["E0", "E1", "E2"],
  "budget": 40,
  "synthesis_cadence": 8,
  "tool_count": 14,
  "step": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Echo of the session ID |
| `enabled_tiers` | string[] | Active representation tiers |
| `budget` | int | Total tool-call budget for this episode |
| `synthesis_cadence` | int | Steps between required syntheses |
| `tool_count` | int | Number of tools available |
| `step` | int | Always `0` on connect |

### 2.2 `tool_result`

Response to a `tool_call` request.

```json
{
  "type": "tool_result",
  "id": "call_1",
  "ok": true,
  "data": {
    "handle": "h_abc12345",
    "summary": {"depth_used": 8}
  },
  "step": 1,
  "budget_remaining": 39,
  "reward_so_far": 0.5,
  "reward_multiplier": 1.0
}
```

When `ok` is `false`, `data` contains `"error"` (error code string) and
`"detail"` (human-readable message):

```json
{
  "type": "tool_result",
  "id": "call_2",
  "ok": false,
  "data": {
    "error": "INVALID_HANDLE",
    "detail": "Handle may have been evicted. Re-encode the value."
  },
  "step": 3,
  "budget_remaining": 37,
  "reward_so_far": 0.0,
  "reward_multiplier": 1.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Echoes the `id` from the `tool_call` |
| `ok` | bool | `true` on success, `false` on tool-level error |
| `data` | object | Tool output (success) or `{"error":…,"detail":…}` (failure) |
| `step` | int | Tool-call count including this one |
| `budget_remaining` | int | Remaining calls before budget exhaustion |
| `reward_so_far` | float | Cumulative episode reward at this step |
| `reward_multiplier` | float | Current synthesis-cadence multiplier (see §4) |

### 2.3 `synthesis_required`

Sent when the tool-call count reaches a synthesis cadence boundary
(`step % synthesis_cadence == 0`). The client **must** respond with a
`synthesis` message; continuing to call tools without synthesising causes the
reward multiplier to decay (see §4).

```json
{
  "type": "synthesis_required",
  "step": 8,
  "budget_remaining": 32,
  "reward_so_far": 2.3,
  "reward_multiplier": 1.0
}
```

### 2.4 `synthesis_scored`

Sent after the server scores a `synthesis` message.

```json
{
  "type": "synthesis_scored",
  "id": "synth_1",
  "reward_after": 2.8,
  "reward_delta": 0.5,
  "conjectures_extracted": 1,
  "conjecture_ids": ["conj_xyz789ab"],
  "prior_relevant": [],
  "reward_multiplier": 1.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Echoes the `id` from the `synthesis` message |
| `reward_after` | float | Total episode reward after scoring |
| `reward_delta` | float | Reward added by this synthesis |
| `conjectures_extracted` | int | Number of conjectures parsed from the synthesis text |
| `conjecture_ids` | string[] | IDs assigned to each extracted conjecture |
| `prior_relevant` | object[] | Previously seen conjectures with high semantic overlap |
| `reward_multiplier` | float | Multiplier reset to `1.0` after a successful synthesis |

### 2.5 `episode_complete`

Sent when the episode ends (either via `episode_end` from the client, or
budget exhaustion).

```json
{
  "type": "episode_complete",
  "total_reward": 3.2,
  "reward_breakdown": {
    "reward_base": 3.2,
    "synthesis_skip_penalty": 0.0
  },
  "steps": 8,
  "syntheses": 1,
  "conjectures_produced": 1,
  "conjectures_board_eligible": 1
}
```

### 2.6 `error`

Sent for protocol-level errors (as opposed to tool-level errors, which arrive
inside `tool_result`).

```json
{
  "type": "error",
  "id": "call_1",
  "code": "INVALID_TOOL",
  "message": "Unknown tool: mc.frobnicate"
}
```

`id` is `null` (normalised to `""` by the client library) when the error is
not tied to a specific request.

#### Error codes

| Code | Trigger |
|------|---------|
| `INVALID_TOOL` | Tool name not recognised |
| `INVALID_MESSAGE` | Malformed JSON or unparseable message type |
| `INVALID_HANDLE` | Handle is expired, evicted, or never existed |
| `UNSUPPORTED_OPERATION` | Operation not supported for the current ladder mode |
| `VIEW_NOT_APPLICABLE` | Requested inspection view not available for this representation |
| `UNDERFLOW` | Normalisation produced a negative decimal (decimal mode only) |

Tool-specific failure codes (carried inside `tool_result.data.error`):
`encode_failed`, `decode_failed`, `zoom_eval_failed`, `capabilities_failed`,
`session_open_failed`, `hist_calibrate_failed`, `score_surprise_failed`,
`arithmetic_failed`, `normalize_failed`, `inspect_failed`.

### 2.7 `ping`

Sent by the server every **30 seconds** as a heartbeat. The client must reply
with a `pong` within **10 seconds** or the connection will be closed with code
`4400` after two consecutive missed pongs.

```json
{"type": "ping"}
```

---

## 3. Client → Server Messages

### 3.1 `tool_call`

```json
{
  "type": "tool_call",
  "id": "call_1",
  "tool": "mc.encode",
  "args": {"value": 42}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Client-assigned correlation ID; echoed in `tool_result` |
| `tool` | string | Tool name (see §5) |
| `args` | object | Tool-specific arguments |

### 3.2 `synthesis`

Submits a free-text conjecture or summary. Required at each synthesis cadence
boundary.

```json
{
  "type": "synthesis",
  "id": "synth_1",
  "text": "Conjecture: all Zeckendorf representations exhibit monotonic digit mass."
}
```

### 3.3 `episode_end`

Gracefully terminates the episode. The server responds with `episode_complete`.

```json
{
  "type": "episode_end",
  "reason": "budget_exhausted"
}
```

Valid reasons: `no_more_conjectures`, `client_stop`, `budget_exhausted`.

### 3.4 `pong`

Reply to a server `ping`. Must be sent within 10 seconds.

```json
{"type": "pong"}
```

---

## 4. Synthesis Cadence and Reward Multiplier Decay

Every `synthesis_cadence` tool steps (default: 8), the server sends
`synthesis_required`. If the client calls more tools without first responding
with a `synthesis`, the **reward multiplier** decays:

```
multiplier = max(0.0, 1.0 − 0.15 × skipped_steps)
```

where `skipped_steps` is the number of tool calls made since the cadence
boundary without a synthesis.

| Skipped steps | Multiplier |
|--------------|-----------|
| 0 | 1.00 |
| 1 | 0.85 |
| 2 | 0.70 |
| 3 | 0.55 |
| 4 | 0.40 |
| 5 | 0.25 |
| 6 | 0.10 |
| 7+ | 0.00 |

The multiplier is applied to all reward earned in that synthesis window.
Submitting a `synthesis` resets the multiplier to `1.0` for the next window.

---

## 5. Available Tools

All tool names use the `mc.` prefix (short aliases without the prefix also
work).

| Tool | Purpose |
|------|---------|
| `mc.encode` | Encode a numeric value into a representation handle |
| `mc.encode_deeper` | Re-encode an existing handle at greater depth |
| `mc.decode` | Decode a handle back to its numeric value |
| `mc.zoom_eval` | Evaluate a handle at a specific zoom depth |
| `mc.capabilities` | Query representation capabilities |
| `mc.session_open` | Create a measurement session |
| `mc.hist_calibrate` | Calibrate a histogram (session required) |
| `mc.score_surprise` | Score novelty against a histogram (session required) |
| `mc.arithmetic` | Perform arithmetic on encoded values |
| `mc.divide_in_family` | Divide values within the same representation family |
| `mc.multiply` | Multiply two encoded values |
| `mc.compare` | Compare two encoded values |
| `mc.normalize` | Normalise a digit sequence |
| `mc.inspect` | Inspect internal representation details |

Handles (`h_xxxxxxxx`) are opaque references returned by encoding tools. They
expire after **1 hour** or when the server's handle store reaches capacity
(10 000 handles, LRU-evicted). An evicted handle returns error code
`INVALID_HANDLE`.

---

## 6. Heartbeat

The server sends a `ping` every 30 seconds. The client must respond with
`pong` within 10 seconds. After **two consecutive missed pongs** the server
closes the connection with WebSocket close code `4400`.

---

## 7. Disconnect and Resume

The protocol does **not** support mid-episode reconnection. Each episode is
bound to a single WebSocket connection. If the connection drops unexpectedly,
the server finalises the episode with reason `"truncated"` and closes the
session. A new episode requires a new `POST /v1/sessions` call.

---

## 8. Typical Episode Flow

```
Client                          Server
  │                               │
  │── POST /v1/sessions ─────────►│
  │◄─ 200 {session_id, ws_url} ───│
  │                               │
  │── WS connect ────────────────►│
  │◄─ session_ready ──────────────│
  │                               │
  │── tool_call (mc.encode) ─────►│
  │◄─ tool_result ────────────────│
  │   ... (repeat up to 8 steps)  │
  │                               │
  │◄─ synthesis_required ─────────│
  │── synthesis ─────────────────►│
  │◄─ synthesis_scored ───────────│
  │                               │
  │   ... (more tool calls) ──────│
  │                               │
  │── episode_end ───────────────►│
  │◄─ episode_complete ───────────│
  │                               │
  │── WS close ──────────────────►│
```
