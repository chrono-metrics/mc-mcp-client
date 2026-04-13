#!/usr/bin/env python3
"""Full GRPO training loop against the MC-MCP gym service.

This example shows the same high-level shape as the private production trainer:

1. Sample one prompt / seed group for the generation
2. Collect G rollouts against the MC-MCP service
3. Read the local JSONL episode log back into chat transcripts
4. Compute GRPO group-relative advantages
5. Hand the rollouts to an external LoRA / RL trainer
6. Hot-reload the updated adapter into vLLM and continue

The reward signal comes directly from `episode_complete.total_reward`, returned
by the service at the end of each episode.

The actual gradient update is framework-specific. The reference production
trainer uses GRPO-style updates as described in DeepSeekMath:
https://arxiv.org/abs/2402.03300

Usage:
  python examples/qwen3_8b_grpo.py --config examples/qwen3_grpo_config.yaml
  python examples/qwen3_8b_grpo.py --config examples/qwen3_grpo_config.yaml --collect-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from mc_mcp_client import (
    DEFAULT_SERVICE_URL,
    EpisodeConfig,
    Orchestrator,
    ServiceConfig,
    SessionConfig,
    VLLMBackend,
)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_DEFAULT_ENABLED_TIERS = ["E0", "E1", "E2"]


@dataclass
class LoRAConfig:
    """Minimal LoRA hyperparameters for the external trainer hook."""

    rank: int = 16
    alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    learning_rate: float = 1.0e-5


@dataclass
class TrainingConfig:
    """Configuration for the public GRPO example."""

    model: str = "Qwen/Qwen3-8B"
    model_url: str = "http://localhost:8080/v1"
    model_path: str = "./models/qwen3-8b"
    adapter_path: str = "./adapters/current"
    service_url: str = DEFAULT_SERVICE_URL
    api_key: str = ""
    stage: int = 2
    G: int = 4
    num_generations: int = 100
    checkpoint_every: int = 10
    local_log_root: str = "./episodes/grpo"
    checkpoint_dir: str = "./checkpoints/grpo"
    random_seed: int | None = None
    seed_count: int = 3
    seed_min: int = 1
    seed_max: int = 100
    enabled_tiers: list[str] = field(default_factory=lambda: list(_DEFAULT_ENABLED_TIERS))
    max_steps: int = 40
    synthesis_cadence: int = 8
    temperature: float = 0.7
    max_tokens: int = 512
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)


async def collect_rollouts(
    orch: Orchestrator,
    *,
    G: int,
    stage: int,
    seeds: list[int],
    log_dir: Path,
) -> list[tuple[list[dict[str, str]], float]]:
    """Run G episodes and return `(messages, reward)` pairs.

    The public orchestrator writes one JSONL file per session. This example
    uses that log as the rollout transcript source so the training loop can
    stay on the public client surface.
    """

    log_dir.mkdir(parents=True, exist_ok=True)
    orch.config.stage = stage
    orch.config.local_log_dir = str(log_dir)

    results = await orch.run_session_async(
        n_episodes=G,
        seeds_per_episode=[list(seeds) for _ in range(G)],
    )

    log_path = _find_session_log(log_dir)
    message_groups = _reconstruct_rollouts_from_log(log_path)
    if len(message_groups) != len(results):
        raise RuntimeError(
            f"Expected {len(results)} rollout transcripts in {log_path}, "
            f"found {len(message_groups)}."
        )

    return [
        (messages, result.total_reward)
        for messages, result in zip(message_groups, results, strict=True)
    ]


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """Normalize rewards within the group for GRPO.

    Group-relative normalization is only meaningful when each rollout in the
    group is answering the same prompt. This example enforces that by reusing
    one seed list across the whole generation.
    """

    if not rewards:
        raise ValueError("Cannot compute GRPO advantages for an empty reward list.")

    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = max(variance**0.5, 1e-6)
    return [(reward - mean) / std for reward in rewards]


async def train(config_path: str, *, collect_only: bool = False) -> None:
    """Run the public GRPO example loop."""

    cfg = load_training_config(config_path)
    rng = random.Random(cfg.random_seed)

    backend = VLLMBackend(model=cfg.model, url=cfg.model_url)
    service = ServiceConfig(service_url=cfg.service_url, api_key=cfg.api_key)
    session = SessionConfig(
        enabled_tiers=list(cfg.enabled_tiers),
        budget=cfg.max_steps,
        synthesis_cadence=cfg.synthesis_cadence,
    )
    episode = EpisodeConfig(
        stage=cfg.stage,
        max_steps=cfg.max_steps,
        synthesis_cadence=cfg.synthesis_cadence,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        local_log_dir=cfg.local_log_root,
    )
    orch = Orchestrator(
        backend=backend,
        service_config=service,
        session_config=session,
        episode_config=episode,
    )

    for generation in range(cfg.num_generations):
        seeds = sample_group_seeds(cfg, rng)
        log_dir = Path(cfg.local_log_root) / f"gen_{generation:04d}"

        rollouts = await collect_rollouts(
            orch,
            G=cfg.G,
            stage=cfg.stage,
            seeds=seeds,
            log_dir=log_dir,
        )
        rewards = [reward for _, reward in rollouts]
        advantages = compute_grpo_advantages(rewards)

        print(
            f"Gen {generation:04d}: seeds={seeds} "
            f"rewards={[f'{reward:.2f}' for reward in rewards]} "
            f"mean={sum(rewards) / len(rewards):.2f}"
        )

        if collect_only:
            print("  collect-only: skipped model update and adapter reload")
        else:
            update_model(
                model_path=cfg.model_path,
                rollouts=[
                    (messages, advantage)
                    for (messages, _), advantage in zip(rollouts, advantages, strict=True)
                ],
                lora_config=cfg.lora_config,
            )
            await reload_model_adapter(backend, cfg.adapter_path)

        if generation % cfg.checkpoint_every == 0:
            checkpoint_path = save_checkpoint(
                generation=generation,
                adapter_path=cfg.adapter_path,
                checkpoint_dir=cfg.checkpoint_dir,
                seeds=seeds,
                rewards=rewards,
                advantages=advantages,
            )
            print(f"  checkpoint metadata: {checkpoint_path}")


def update_model(
    model_path: str,
    rollouts: list[tuple[list[dict[str, str]], float]],
    lora_config: LoRAConfig,
) -> None:
    """Hook point for the actual GRPO / LoRA optimizer step.

    Inputs:
    - `rollouts`: `(messages, advantage)` pairs reconstructed from the client
      episode logs.
    - `lora_config`: rank / alpha / target modules / learning rate.

    Integrate this function with TRL's `GRPOTrainer`, NeMo Aligner, or your own
    training code. The public client intentionally stops at episode collection
    and reward extraction.
    """

    raise NotImplementedError(
        "Integrate update_model() with your RL framework, or run with --collect-only."
    )


async def reload_model_adapter(backend: VLLMBackend, adapter_path: str) -> None:
    """Hook point for reloading a freshly updated LoRA adapter into the server."""

    raise NotImplementedError(
        "Integrate reload_model_adapter() with your serving stack, or run with --collect-only."
    )


def load_training_config(path: str) -> TrainingConfig:
    """Load the example's flat YAML config and expand `${ENV}` references."""

    raw = _load_yaml_mapping(Path(path))
    expanded = _expand_env_vars(raw)

    config_kwargs = _section_kwargs(TrainingConfig, expanded)
    config_kwargs.pop("lora_config", None)
    lora_kwargs = _section_kwargs(LoRAConfig, expanded.get("lora_config"))

    api_key = os.getenv("MC_MCP_API_KEY")
    service_url = os.getenv("MC_MCP_SERVICE_URL")
    model_url = os.getenv("MC_MCP_MODEL_URL")

    if api_key is not None:
        config_kwargs["api_key"] = api_key
    if service_url is not None:
        config_kwargs["service_url"] = service_url
    if model_url is not None:
        config_kwargs["model_url"] = model_url

    cfg = TrainingConfig(
        **config_kwargs,
        lora_config=LoRAConfig(**lora_kwargs),
    )

    if not cfg.api_key.strip():
        raise ValueError("Set MC_MCP_API_KEY or provide api_key in the YAML config.")
    if cfg.G <= 1:
        raise ValueError("G must be at least 2 for group-relative normalization.")
    if cfg.num_generations <= 0:
        raise ValueError("num_generations must be positive.")
    if cfg.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")
    if cfg.seed_count <= 0:
        raise ValueError("seed_count must be positive.")
    if cfg.seed_max < cfg.seed_min:
        raise ValueError("seed_max must be greater than or equal to seed_min.")

    population = cfg.seed_max - cfg.seed_min + 1
    if population < cfg.seed_count:
        raise ValueError(
            "Seed range is too small for the requested seed_count "
            f"({cfg.seed_min}..{cfg.seed_max}, seed_count={cfg.seed_count})."
        )

    return cfg


def sample_group_seeds(cfg: TrainingConfig, rng: random.Random) -> list[int]:
    """Sample one seed list to reuse across the whole GRPO group."""

    return sorted(rng.sample(range(cfg.seed_min, cfg.seed_max + 1), k=cfg.seed_count))


def save_checkpoint(
    *,
    generation: int,
    adapter_path: str,
    checkpoint_dir: str,
    seeds: list[int],
    rewards: list[float],
    advantages: list[float],
) -> Path:
    """Write lightweight checkpoint metadata for the external trainer output."""

    target_dir = Path(checkpoint_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"generation_{generation:04d}.json"
    payload = {
        "generation": generation,
        "adapter_path": adapter_path,
        "seeds": list(seeds),
        "rewards": list(rewards),
        "advantages": list(advantages),
    }
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target_path


def _find_session_log(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("*.jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one session log in {log_dir}, found {len(candidates)}."
        )
    return candidates[0]


def _reconstruct_rollouts_from_log(log_path: Path) -> list[list[dict[str, str]]]:
    """Rebuild per-episode chat transcripts from the orchestrator JSONL log."""

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    episodes: list[list[dict[str, str]]] = []
    current_messages: list[dict[str, str]] | None = None

    for event in events:
        event_type = event.get("event_type")

        if event_type == "episode_start":
            current_messages = [
                {"role": "system", "content": _build_logged_system_prompt(event)},
                {"role": "user", "content": _build_logged_seed_prompt(event)},
            ]
            continue

        if current_messages is None:
            continue

        if event_type == "tool_step":
            assistant_payload = {
                "tool": event.get("tool_name", ""),
                "args": event.get("arguments", {}),
            }
            current_messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(assistant_payload, sort_keys=True),
                }
            )
            current_messages.append(
                {
                    "role": "user",
                    "content": str(event.get("observation_card", "")),
                }
            )
            continue

        if event_type == "synthesis":
            prompt = str(event.get("prompt", "")).strip()
            response = str(
                event.get("response") or event.get("synthesis_text") or ""
            ).strip()
            if prompt:
                current_messages.append({"role": "user", "content": prompt})
            if response:
                current_messages.append({"role": "assistant", "content": response})
            current_messages.append(
                {
                    "role": "user",
                    "content": _build_logged_synthesis_feedback(event),
                }
            )
            continue

        if event_type == "episode_end":
            episodes.append(current_messages)
            current_messages = None

    return episodes


def _build_logged_system_prompt(event: dict[str, Any]) -> str:
    family_name = event.get("family_display_name") or "mathematical"
    stage = event.get("curriculum_stage", 2)
    cadence = event.get("synthesis_cadence", 8)
    budget = event.get("budget", 40)
    enabled_tiers = ", ".join(event.get("enabled_tiers") or [])
    return (
        f"You are a mathematical discovery agent exploring the {family_name} "
        f"representation family. Curriculum stage={stage}. Enabled tiers={enabled_tiers}. "
        f"Budget={budget} tool calls. Synthesis cadence={cadence}. "
        "The transcript below was reconstructed from the MC-MCP client episode log."
    )


def _build_logged_seed_prompt(event: dict[str, Any]) -> str:
    seeds = event.get("seed_integers") or []
    seed_text = ", ".join(str(seed) for seed in seeds) if seeds else "server-selected"
    return (
        "Start the discovery episode.\n"
        f"Seed integers: {seed_text}\n"
        "On tool turns, respond with JSON tool calls. "
        "When synthesis is requested, respond with one falsifiable conjecture or plan."
    )


def _build_logged_synthesis_feedback(event: dict[str, Any]) -> str:
    server_response = event.get("server_response")
    if not isinstance(server_response, dict):
        server_response = {}

    conjecture_ids = server_response.get("conjecture_ids") or event.get("conjecture_ids") or []
    prior_relevant = server_response.get("prior_relevant") or []
    prior_text = ", ".join(
        str(item.get("id", "?")) for item in prior_relevant if isinstance(item, dict)
    ) or "none"
    conjecture_text = ", ".join(str(item) for item in conjecture_ids) or "-"

    return (
        "Synthesis scored.\n"
        f"reward_delta={server_response.get('reward_delta', 0.0)} "
        f"reward_after={server_response.get('reward_after', 0.0)}\n"
        "conjectures_extracted="
        f"{server_response.get('conjectures_extracted', 0)} "
        f"conjecture_ids={conjecture_text}\n"
        f"prior_relevant={prior_text} "
        f"reward_multiplier={event.get('reward_multiplier', 1.0)}"
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Training config must contain a top-level mapping.")
    return data


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _section_kwargs(cls: type[Any], section: Any) -> dict[str, Any]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section for {cls.__name__} must be a mapping.")
    allowed_keys = {item.name for item in fields(cls)}
    return {key: value for key, value in section.items() if key in allowed_keys}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="examples/qwen3_grpo_config.yaml",
        help="Path to the GRPO example YAML config.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect rollouts and compute GRPO advantages, but skip the trainer hooks.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(train(args.config, collect_only=args.collect_only))


if __name__ == "__main__":
    main()
