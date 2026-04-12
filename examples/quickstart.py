#!/usr/bin/env python3
"""examples/quickstart.py - Run three episodes in one session against the MC-MCP gym."""

from __future__ import annotations

import asyncio

from mc_mcp_client import EpisodeConfig, Orchestrator, SessionConfig, VLLMBackend


async def main() -> None:
    # 1. Connect to your model server (vLLM, Ollama, TGI, etc.)
    backend = VLLMBackend(
        model="qwen3-8b",
        base_url="http://localhost:8080/v1",  # <- your model server
    )

    # 2. Configure the hosted curriculum session
    session = SessionConfig(
        enabled_tiers=["E0", "E1", "E2"],
    )

    episode = EpisodeConfig(
        stage=2,
        max_steps=40,
    )

    if not await backend.check_health():
        raise RuntimeError(f"Model server is not reachable at {backend.base_url}")

    orch = Orchestrator(
        backend=backend,
        api_key="mcmcp_dev_key",             # <- your API key
        session_config=session,
        service_url="ws://localhost:9090",   # <- optional local/self-hosted override
        episode_config=episode,
    )

    # 3. Run 3 episodes in one session (surprise histograms warm up across episodes)
    results = await orch.run_session(
        n_episodes=3,
        seeds_per_episode=[[17, 23], [42, 55], [89, 144]],
    )

    # 4. See what happened
    for i, r in enumerate(results):
        print(
            f"Episode {i + 1}: reward={r.total_reward:.2f} "
            f"conjectures={r.conjectures_produced} "
            f"board_eligible={r.conjectures_board_eligible}"
        )

    total = sum(r.total_reward for r in results)
    print(f"\nSession total reward: {total:.2f}")
    print(f"Episode logs: {episode.local_log_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
