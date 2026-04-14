# mc-mcp-client

Thin client for running episodes against the MC-MCP curriculum service.

### Which URL should I use?

> By default, the client targets the hosted Chrono Metrics service at `ws://gym.chrono-metrics.com/v1/sessions/{session_id}/ws`. Override `MC_MCP_SERVICE_URL` only for local or self-hosted deployments.

| Situation | What to set |
|---|---|
| Using hosted service | `MC_MCP_API_KEY` |
| Running local server/proxy | `MC_MCP_API_KEY`, `MC_MCP_SERVICE_URL=ws://localhost:9090` |
| Using your own deployment | `MC_MCP_API_KEY`, `MC_MCP_SERVICE_URL=...` |

## Quickstart (hosted)

```bash
pip install mc-mcp-client[vllm]
vllm serve Qwen/Qwen3-8B --port 8080
export MC_MCP_API_KEY=...
python examples/quickstart.py
```

## Quickstart (local/dev)

```bash
pip install mc-mcp-client[vllm]
vllm serve Qwen/Qwen3-8B --port 8080
export MC_MCP_API_KEY=...
export MC_MCP_SERVICE_URL=ws://localhost:9090
python examples/quickstart.py
```

The example uses exactly this public contract:

```python
import os

from mc_mcp_client import DEFAULT_SERVICE_URL, Orchestrator, SessionConfig, VLLMBackend

backend = VLLMBackend(
    model="qwen3-8b",
    url=os.getenv("MC_MCP_MODEL_URL", "http://localhost:8080"),
)

orch = Orchestrator(
    backend=backend,
    api_key=os.environ["MC_MCP_API_KEY"],
    service_url=os.getenv("MC_MCP_SERVICE_URL", DEFAULT_SERVICE_URL),
    session_config=SessionConfig(enabled_tiers=["E0", "E1", "E2"]),
)

result = orch.run_episode(seeds=[17, 23, 42])
```

Notes:

- `VLLMBackend(url=...)` appends `/v1` automatically. `base_url=.../v1` is still supported if you need it.
- `run_episode(...)` is the beginner-facing synchronous entrypoint.
- Use `run_episode_async(...)` or `run_session_async(...)` only if you are already inside asyncio code.

### Common setup mistakes

- **`MC_MCP_API_KEY` not set** — the client will exit immediately with an error. Make sure the variable is exported in your shell, not just defined in a dotfile that hasn't been sourced.
- **Pointing at `localhost` without a local server running** — if you set `MC_MCP_SERVICE_URL=ws://localhost:9090` but nothing is listening there, the connection will hang or fail. Only set this when you have a local server or `gcloud run services proxy` active.
- **Forgetting to set `MC_MCP_SERVICE_URL` for self-hosted infra** — without the override the client connects to the hosted service, not your deployment. Check the startup log line (`Using MC-MCP service: ...`) to confirm.
- **Wrong protocol (`ws://` vs `wss://`)** — use `ws://` for local/unencrypted and `wss://` for TLS-terminated endpoints. A mismatch will cause a connection error or TLS handshake failure.

## Examples

- `examples/quickstart.py` — minimal single-episode example
