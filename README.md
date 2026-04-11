# mc-mcp-client

Reference client for the [MC-MCP](https://github.com/YOUR_ORG/MC-MCP)
mathematical reasoning gym.

## Install

```
pip install mc-mcp-client
```

## Quickstart

```python
from mc_mcp_client import Orchestrator, VLLMBackend

orch = Orchestrator(
    backend=VLLMBackend(model="qwen3-8b", base_url="http://localhost:8080/v1"),
    service_url="ws://localhost:9090",
    api_key="mcmcp_dev_key",
)
result = orch.run_episode(seeds=[17, 23, 42], stage=2)
print(f"Reward: {result.total_reward}")
```

## Examples

- `examples/quickstart.py` — minimal single episode
- `examples/qwen3_8b_grpo.py` — full RL training loop with GRPO

## Protocol

See [docs/protocol.md](docs/protocol.md) for the WebSocket protocol spec.
