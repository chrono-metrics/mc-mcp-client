# HOTFIX-01e Smoke Test Results

**Date:** 2026-04-12  
**Service:** Local (`ws://localhost:9092`)  
**Service version:** `0.1.0`  
**Client:** `mc-mcp-client` commit `08613ab` (HOTFIX-01d)  
**Family config:** `{"mode": "zeckendorf", "depth": 20}`  
**Model:** Scripted backend (see note)  

> **Model note:** Qwen3-8B is not available in the test environment.
> Tests 1, 2, and 4 used a `ScriptedBackend` that sends one `mc.encode` call
> then stops — sufficient to verify the full protocol path end-to-end.
> Tests 3 and 5 require a real model and manual execution (see §Status below).

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
MC_MCP_SERVICE_URL=ws://localhost:9092 MC_MCP_API_KEY=mcmcp_dev

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
