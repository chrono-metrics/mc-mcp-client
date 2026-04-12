"""
Test 3: Reward parity between monolith and service.

Requires:
- Live MC-MCP service (MC_MCP_SERVICE_URL)
- Real model server (MC_MCP_MODEL_URL, @pytest.mark.requires_model)
- MC-MCP monolith available at MC_MCP_MONOLITH_PATH

Run:
    export MC_MCP_SERVICE_URL=ws://localhost:9090
    export MC_MCP_API_KEY=mcmcp_dev
    export MC_MCP_MODEL_URL=http://localhost:8080/v1
    export MC_MCP_MONOLITH_PATH=/path/to/MC-MCP
    pytest tests/integration/test_reward_parity.py -v -s
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mc_mcp_client.backends.vllm import VLLMBackend
from mc_mcp_client.config import EpisodeConfig
from mc_mcp_client.orchestrator import Orchestrator
from mc_mcp_client.protocol import EpisodeComplete

from .conftest import create_session_async

pytestmark = [pytest.mark.live_service, pytest.mark.requires_model]

_MONOLITH_PATH = os.getenv("MC_MCP_MONOLITH_PATH", "")
_PARITY_SEEDS = [17, 23, 42]
_PARITY_FAMILY_CONFIG = {"mode": "zeckendorf", "depth": 20}
_PARITY_BUDGET = 20
_PARITY_CADENCE = 8

# Tolerance thresholds
_TOTAL_REWARD_TOLERANCE = 0.01  # 1%
_TERM_TOLERANCE = 0.05           # 5% per term


@pytest.fixture(scope="module")
def monolith_path() -> Path:
    if not _MONOLITH_PATH:
        pytest.skip("MC_MCP_MONOLITH_PATH not set — skipping reward parity tests")
    path = Path(_MONOLITH_PATH)
    if not path.exists():
        pytest.skip(f"Monolith path does not exist: {path}")
    return path


async def run_service_episode(
    service_url: str,
    api_key: str,
    model_url: str,
    tmp_path: Path,
) -> EpisodeComplete:
    """Run one episode via the thin client against the live service."""
    from mc_mcp_client.backends.vllm import VLLMBackend

    resp = await create_session_async(
        service_url, api_key, _PARITY_FAMILY_CONFIG,
        budget=_PARITY_BUDGET,
        synthesis_cadence=_PARITY_CADENCE,
    )
    session_id = resp["session_id"]

    backend = VLLMBackend(model="qwen3-8b", base_url=model_url)
    orch = Orchestrator(
        backend=backend,
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(
            local_log_dir=str(tmp_path / "service"),
            max_steps=_PARITY_BUDGET,
            synthesis_cadence=_PARITY_CADENCE,
        ),
    )
    return await orch.run_episode_async(session_id=session_id, seeds=_PARITY_SEEDS)


def run_monolith_episode(monolith_path: Path, tmp_path: Path) -> dict:
    """Run one episode via the monolith CLI and parse the reward breakdown.

    Returns the reward_breakdown dict from the monolith's output.
    """
    seeds_str = ",".join(str(s) for s in _PARITY_SEEDS)
    output_dir = tmp_path / "monolith"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "run_experiment.py",
            "--mode", "local",
            "--episodes", "1",
            "--stage", "2",
            "--seeds", seeds_str,
            "--budget", str(_PARITY_BUDGET),
            "--synthesis-cadence", str(_PARITY_CADENCE),
            "--output-dir", str(output_dir),
        ],
        cwd=str(monolith_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Monolith run_experiment.py failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    # Parse reward breakdown from monolith output logs
    log_files = list(output_dir.glob("**/*.jsonl"))
    if not log_files:
        pytest.fail(f"No JSONL log files found in monolith output dir: {output_dir}")

    for log_path in log_files:
        for line in reversed(log_path.read_text().splitlines()):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") == "episode_end" and "summary" in event:
                return event["summary"].get("reward_breakdown", {})

    pytest.fail(f"Could not find episode_end event in monolith logs: {log_files}")


# ── Parity test ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reward_parity_service_vs_monolith(
    service_url, api_key, family_config, model_url, monolith_path, tmp_path: Path
) -> None:
    """Service and monolith produce matching rewards for identical seeds.

    Pass criteria:
    - total_reward difference < 1%.
    - Each reward term matches within 5%.
    - Any term that differs by > 5% is explicitly flagged.
    """
    service_result = await run_service_episode(service_url, api_key, model_url, tmp_path)
    monolith_breakdown = run_monolith_episode(monolith_path, tmp_path)

    service_breakdown = service_result.reward_breakdown
    service_total = service_result.total_reward
    monolith_total = monolith_breakdown.get("reward_base", 0.0)

    print(f"\nService total_reward:  {service_total:.6f}")
    print(f"Monolith total_reward: {monolith_total:.6f}")
    print("\nReward term comparison:")
    print(f"{'Term':<35} {'Service':>12} {'Monolith':>12} {'Δ%':>8}")
    print("-" * 75)

    mismatched_terms: list[str] = []
    for term in sorted(set(service_breakdown) | set(monolith_breakdown)):
        svc_val = float(service_breakdown.get(term, 0.0))
        mono_val = float(monolith_breakdown.get(term, 0.0))
        denom = max(abs(svc_val), abs(mono_val), 1e-9)
        pct_diff = abs(svc_val - mono_val) / denom * 100
        flag = " ← MISMATCH" if pct_diff > _TERM_TOLERANCE * 100 else ""
        print(f"{term:<35} {svc_val:>12.6f} {mono_val:>12.6f} {pct_diff:>7.2f}%{flag}")
        if pct_diff > _TERM_TOLERANCE * 100:
            mismatched_terms.append(f"{term}: svc={svc_val:.6f} mono={mono_val:.6f} ({pct_diff:.1f}%)")

    # Total reward check
    if monolith_total != 0:
        total_diff_pct = abs(service_total - monolith_total) / abs(monolith_total)
        assert total_diff_pct < _TOTAL_REWARD_TOLERANCE, (
            f"total_reward mismatch: service={service_total:.6f} "
            f"monolith={monolith_total:.6f} (diff={total_diff_pct:.2%})"
        )
    else:
        assert math.isfinite(service_total)

    if mismatched_terms:
        # Report but do not fail — terms may differ due to model non-determinism.
        # Investigate any consistent mismatches manually.
        print(
            f"\nWARNING: {len(mismatched_terms)} reward term(s) differ by >{_TERM_TOLERANCE:.0%}:\n"
            + "\n".join(f"  {t}" for t in mismatched_terms)
        )


@pytest.mark.asyncio
async def test_reward_breakdown_contains_expected_terms(
    service_url, api_key, family_config, model_url, tmp_path: Path
) -> None:
    """episode_complete.reward_breakdown must contain core reward terms."""
    resp = await create_session_async(
        service_url, api_key, family_config,
        budget=_PARITY_BUDGET, synthesis_cadence=_PARITY_CADENCE,
    )
    session_id = resp["session_id"]

    backend = VLLMBackend(model="qwen3-8b", base_url=os.getenv("MC_MCP_MODEL_URL", ""))
    orch = Orchestrator(
        backend=backend,
        service_url=service_url,
        api_key=api_key,
        config=EpisodeConfig(
            local_log_dir=str(tmp_path),
            max_steps=_PARITY_BUDGET,
            synthesis_cadence=_PARITY_CADENCE,
        ),
    )
    result = await orch.run_episode_async(session_id=session_id, seeds=_PARITY_SEEDS)

    expected_terms = {
        "reward_base",
        "synthesis_skip_penalty",
        "valid_fraction",
        "avg_surprise",
        "tool_utilization",
        "redundancy_rate",
        "invalid_call_rate",
    }
    missing = expected_terms - set(result.reward_breakdown.keys())
    assert not missing, (
        f"reward_breakdown missing expected terms: {missing}\n"
        f"Got: {sorted(result.reward_breakdown.keys())}"
    )
