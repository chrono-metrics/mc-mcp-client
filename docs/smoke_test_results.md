# HOTFIX-01e Smoke Test Results

**Date:** 2026-04-12  
**Service:** Cloud Run (`https://mc-mcp-ws-service-687522919138.us-central1.run.app`)  
**Service revision:** `mc-mcp-ws-service-00003-852` (HOTFIX-01c deployed 2026-04-12)  
**Access:** `gcloud run services proxy mc-mcp-ws-service --project=ladder-gym --region=us-central1 --port=9091`  
**Client:** `mc-mcp-client` commit `d2d8899` (HOTFIX-01e)  
**Family config:** `{"mode": "zeckendorf", "depth": 20}`  
**Model:** Scripted backend for the automated smoke suite; `qwen3:8b` via Ollama on `grpo-l4-runner` for the manual live run below  

> **Environment note:** direct requests to the Cloud Run URL returned `401 Unauthorized`
> without gcloud-authenticated proxying, so all live service runs used
> `gcloud run services proxy ... --port=9091`.
> The available L4 runner was `grpo-l4-runner` in `us-east1-b` rather than
> `us-central1-a`; it initially had no external IP, so a temporary access config
> was added to download Ollama and `qwen3:8b`.
> Tests 1, 2, and 4 below still use a `ScriptedBackend`; the real-model findings
> are recorded in the addendum below.

---

## Test 1: Single episode, verify reward

**Procedure:** `pytest tests/integration/test_e2e_smoke.py::test_single_episode_completes_with_reward`

**Result: PASS**

```
session_id:          02be4148a768
family_display_name: Zeckendorf (Fibonacci-indexed, depth 20)

Episode summary
---------------
total_reward:              3.275
steps:                     1
syntheses:                 0
conjectures_produced:      0
conjectures_board_eligible: 0

reward_breakdown (16 terms)
---------------------------
avg_surprise:              0.0
branch_drift_rate:         0.0
error_recovery_rate:       0.5
focus_after_synth:         0.0
fresh_test_rate:           1.0
invalid_call_rate:         0.0
protocol_recovery_reward:  0.0
qualified_clean_stop:      0.0
redundancy_rate:           0.0
repeated_error_rate:       0.0
result_followup_rate:      0.0
reward_base:               3.275
synth_score:               0.0
synthesis_skip_penalty:    -0.0
tool_utilization:          0.05
valid_fraction:            1.0
```

**Verification:**
- ✅ Episode completed without errors.
- ✅ `tool_result` for `mc.encode` returned `ok=true` with handle `h_xxxxxxxx` (no config errors).
- ✅ `total_reward` is a finite float (`3.275`).
- ✅ `reward_breakdown` contains all 16 terms.
- ✅ Episode log written to `./episodes/<session_id>.jsonl`.

---

## Test 2: Multi-episode session, histogram persistence

**Procedure:** `pytest tests/integration/test_e2e_smoke.py::test_multi_episode_session`

**Result: PASS**

- 3 episodes completed in one WebSocket connection.
- Each episode returned a finite `total_reward`.
- Episode log accumulated 3 `episode_start` events in the session file.
- `test_handle_from_episode1_invalid_in_episode2` — **PASS**: handle used in
  episode 2 produced a failed tool step (expected).

> **Histogram observation:** `avg_surprise` is `0.0` for the scripted backend
> (no `mc.score_surprise` calls). To verify surprise accumulation across
> episodes, re-run with a real model that calls `mc.score_surprise` in episodes
> 2+. Expected behaviour: episode 1 has no histogram calibration; episodes 2+
> reflect prior calibration.

---

## Test 3: Reward parity vs monolith

**Status: PENDING — requires real model + monolith**

**To run:**
```bash
export MC_MCP_SERVICE_URL=ws://localhost:9090
export MC_MCP_API_KEY=mcmcp_dev
export MC_MCP_MODEL_URL=http://localhost:8080/v1
export MC_MCP_MONOLITH_PATH=/path/to/MC-MCP
pytest tests/integration/test_reward_parity.py -v -s
```

**Expected pass criteria:**
- `total_reward` difference < 1%.
- All 16 reward terms within 5%.
- `reward_breakdown` contains the same set of terms.

---

## Addendum: Manual live run with real Qwen3

**Procedure:**

1. Start authenticated Cloud Run proxy locally:
   ```bash
   gcloud run services proxy mc-mcp-ws-service \
       --project=ladder-gym --region=us-central1 --port=9091
   ```
2. Use `grpo-l4-runner` in `us-east1-b` to host Ollama with `qwen3:8b`.
3. Tunnel model port locally:
   ```bash
   gcloud compute ssh grpo-l4-runner --project=ladder-gym --zone=us-east1-b \
       -- -N -L 11434:127.0.0.1:11434
   ```
4. Run the thin client against:
   - service: `ws://127.0.0.1:9091`
   - model: `http://127.0.0.1:11434/v1`

### Run A: stock thin-client prompt

**Session:** `b4379fd55c78`  
**Result:** PARTIAL — transport succeeded, but the then-current thin client
ended the episode as `parse_error`

- Session creation succeeded with family info and websocket connect succeeded.
- `VLLMBackend.check_health()` passed against the tunneled Ollama endpoint.
- The episode completed end-to-end with a server `episode_complete`, but with
  `0` tool steps and `total_reward=1.75`.
- Log: `episodes/live_ollama_qwen3/b4379fd55c78.jsonl`

**Observed client compatibility gap:**

The local Qwen3/Ollama stack answered the first turn with multiple JSON tool
calls in a single assistant message:

```json
{"tool": "mc.encode", "args": {"n": 17}}
{"tool": "mc.encode", "args": {"n": 23}}
{"tool": "mc.encode", "args": {"n": 42}}
```

This is not a service-side bug and not an unexpected model defect. It matches
the monolith design, which allowed same-turn batches for a small whitelist of
batchable tools and then concatenated the resulting observation cards. The
then-current thin client still enforced a stricter one-call-per-turn parser, so
it rejected this monolith-compatible `mc.encode` batch and ended the episode
with `parse_error`.

### Run B: one-off stricter prompt override

**Session:** `c9b6772f3f58`  
**Result:** PASS — real episode completed against the live websocket service

```
steps:         4
syntheses:     0
total_reward:  3.1625
valid_fraction: 0.75
invalid_call_rate: 0.25
```

**Tool trace:**

1. `mc.encode(n=17)` → `ok=true`, handle returned
2. `mc.compare(lhs=h_80963163, rhs=23)` → invalid call from model
3. `mc.encode(n=23)` → `ok=true`, handle returned
4. `mc.encode(n=42)` → `ok=true`, handle returned

**Verification:**

- ✅ Live Cloud Run websocket path worked through the thin client.
- ✅ Real model responses reached the service and produced real `tool_result` messages.
- ✅ No `config` field was supplied by the client; server-side injection still worked.
- ✅ Episode completed with finite `total_reward`.
- ✅ Local log written to `episodes/live_ollama_qwen3_strict/c9b6772f3f58.jsonl`.

**Finding:**

The websocket service path is healthy. Run A exposed a thin-client
compatibility regression relative to the monolith conversation loop: the client
needed to accept same-turn batches for batchable tools such as `mc.encode`
rather than rejecting them. The service protocol did not need any change.
Tightening the prompt was a temporary workaround for the old client behavior,
but the real fix belongs in the client parser/orchestrator so monolith-style
batchable turns work again.

---

## Test 4: Model never sends config

**Procedure:** `pytest tests/integration/test_e2e_smoke.py::test_tool_calls_contain_no_config_field`

**Result: PASS**

```bash
# Equivalent manual check:
grep '"config"' episodes/*.jsonl
# (no output — zero matches)
```

- ✅ Zero tool call args contained a `config` key across the entire episode log.
- ✅ System prompt contains the explicit instruction
  `"do NOT include a 'config' field"`.
- ✅ All tool calls succeeded without config (server injected family config).

---

## Test 5: Error recovery (disconnect mid-episode)

**Status: PENDING — requires manual execution**

**Procedure:**
1. Start the service.
2. Start an episode (`python examples/quickstart.py` with model running).
3. Kill the service process (`kill <pid>`) mid-episode.
4. Observe client behaviour.

**Expected:**
- Client raises `ConnectionError` from `recv()` within ping timeout (≤ 30 s).
- Episode log on disk contains at least `episode_start` event.
- After service restart, a new session starts cleanly.

---

## Integration test suite run

```
pytest tests/integration/test_e2e_smoke.py -v
MC_MCP_SERVICE_URL=ws://localhost:9091 MC_MCP_API_KEY=mcmcp_dev  (via gcloud proxy)

PASSED  test_service_health
PASSED  test_session_create_returns_family_info
PASSED  test_websocket_session_ready_contains_family_info
PASSED  test_single_episode_completes_with_reward
PASSED  test_tool_result_has_valid_handle
PASSED  test_multi_episode_session
PASSED  test_handle_from_episode1_invalid_in_episode2
PASSED  test_tool_calls_contain_no_config_field
PASSED  test_system_prompt_instructs_no_config
SKIPPED test_full_episode_with_qwen3  [MC_MCP_MODEL_URL not set]

9 passed, 1 skipped in 0.69s
```

**Manual follow-up:**

- Real-model run executed outside pytest using the live Cloud Run proxy and
  Ollama-hosted `qwen3:8b`.
- Stock prompt run completed but stopped as `parse_error`.
- Stricter prompt override completed a real 4-step episode successfully.

---

## Service access for Cloud Run

**Recommended (no public exposure):**
```bash
# Tunnels through gcloud credentials — no IAM changes needed.
gcloud run services proxy mc-mcp-ws-service \
    --project=ladder-gym --region=us-central1 --port=9090
export MC_MCP_SERVICE_URL=ws://localhost:9090
```

**Alternative (public, lower security):**
```bash
gcloud run services update mc-mcp-ws-service \
    --region=us-central1 --ingress=all --no-invoker-iam-check
```
> ⚠️ This exposes the service to anyone with the URL + API key.
> The default key `mcmcp_dev` is well-known. Acceptable for short-term dev
> testing; revert with `--ingress=internal` when done.
