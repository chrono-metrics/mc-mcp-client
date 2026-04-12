# mc-mcp-client

Reference client for the [MC-MCP](https://github.com/YOUR_ORG/MC-MCP)
mathematical reasoning gym.

## Getting Started

1. Install:
   `pip install mc-mcp-client`
2. Start your model:
   `vllm serve Qwen/Qwen3-8B --port 8080`
3. Start the gym:
   `docker compose up`
   Run this from the [MC-MCP](https://github.com/YOUR_ORG/MC-MCP) repo.
4. Edit `examples/quickstart.py` with your model URL and API key, then run:
   `python examples/quickstart.py`
5. See your episode log in `./episodes/`

The public `VLLMBackend` example uses:

```python
backend = VLLMBackend(model="qwen3-8b", url="http://localhost:8080")
```

The client normalizes that to the OpenAI-compatible `/v1` base path internally.
`base_url="http://localhost:8080/v1"` is still supported as an explicit compatibility alias.

The beginner-facing runtime API is synchronous:

```python
result = orch.run_episode(seeds=[17, 23, 42])
```

If you are already inside asyncio code, use the async variants instead:

```python
result = await orch.run_episode_async(seeds=[17, 23, 42])
results = await orch.run_session_async(n_episodes=3, seeds_per_episode=[[17, 23], [42, 55], [89, 144]])
```

## Examples

- `examples/quickstart.py` — minimal single episode
- `examples/qwen3_8b_grpo.py` — full RL training loop with GRPO

## Protocol

See [docs/protocol.md](docs/protocol.md) for the WebSocket protocol spec.
