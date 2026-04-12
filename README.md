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

## Examples

- `examples/quickstart.py` — minimal single episode
- `examples/qwen3_8b_grpo.py` — full RL training loop with GRPO

## Protocol

See [docs/protocol.md](docs/protocol.md) for the WebSocket protocol spec.
