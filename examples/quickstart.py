#!/usr/bin/env python3
"""Run one episode with the public thin-client contract."""

from __future__ import annotations

import os
import sys

from mc_mcp_client import DEFAULT_SERVICE_URL, Orchestrator, SessionConfig, VLLMBackend


def main() -> None:
    api_key = os.getenv("MC_MCP_API_KEY", "").strip()
    if not api_key:
        print("Set MC_MCP_API_KEY environment variable", file=sys.stderr)
        raise SystemExit(1)

    model_url = os.getenv("MC_MCP_MODEL_URL", "http://localhost:8080")
    service_url = os.getenv("MC_MCP_SERVICE_URL", DEFAULT_SERVICE_URL)

    print(f"Using MC-MCP service: {service_url}")

    backend = VLLMBackend(
        model="qwen3-8b",
        url=model_url,
    )
    orch = Orchestrator(
        backend=backend,
        api_key=api_key,
        service_url=service_url,
        session_config=SessionConfig(enabled_tiers=["E0", "E1", "E2"]),
    )

    result = orch.run_episode(seeds=[17, 23, 42])

    print(
        f"Episode reward={result.total_reward:.2f} "
        f"conjectures={result.conjectures_produced} "
        f"board_eligible={result.conjectures_board_eligible}"
    )


if __name__ == "__main__":
    main()
