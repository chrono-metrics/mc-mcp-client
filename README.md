# mc-mcp-client

Thin client for running episodes against the MC-MCP curriculum service.

## Quickstart

1. Install the client:
   `pip install mc-mcp-client[vllm]`
2. Start an OpenAI-compatible model server:
   `vllm serve Qwen/Qwen3-8B --port 8080`
3. Provide your MC-MCP API key:
   `export MC_MCP_API_KEY=...`
4. Optionally override the curriculum service URL:
   `export MC_MCP_SERVICE_URL=ws://localhost:9090`
5. Optionally override the model URL:
   `export MC_MCP_MODEL_URL=http://localhost:8080`
6. Run the example:
   `python examples/quickstart.py`

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

## Examples

- `examples/quickstart.py` — minimal single-episode example
