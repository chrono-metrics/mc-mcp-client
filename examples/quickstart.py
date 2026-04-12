#!/usr/bin/env python3
"""examples/quickstart.py - Run one episode against the MC-MCP gym."""

from __future__ import annotations

from mc_mcp_client import EpisodeConfig, Orchestrator, SessionConfig, VLLMBackend


def main() -> None:
    # 1. Connect to your model server (vLLM, Ollama, TGI, etc.)
    backend = VLLMBackend(
        model="qwen3-8b",
        url="http://localhost:8080",  # <- your model server; /v1 is appended automatically
    )

    # 2. Configure the hosted curriculum session
    session = SessionConfig(
        enabled_tiers=["E0", "E1", "E2"],
    )

    episode = EpisodeConfig(
        stage=2,
        max_steps=40,
    )

    orch = Orchestrator(
        backend=backend,
        api_key="mcmcp_dev_key",             # <- your API key
        session_config=session,
        service_url="ws://localhost:9090",   # <- optional local/self-hosted override
        episode_config=episode,
    )

    # 3. Run one episode with seeds, no await required.
    result = orch.run_episode(
        seeds=[17, 23, 42],
    )

    # 4. See what happened
    print(
        f"Episode reward={result.total_reward:.2f} "
        f"conjectures={result.conjectures_produced} "
        f"board_eligible={result.conjectures_board_eligible}"
    )
    print(f"Episode logs: {episode.local_log_dir}/")


if __name__ == "__main__":
    main()
