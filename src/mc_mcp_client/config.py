"""Configuration loading for the MC-MCP client."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class EpisodeConfig:
    """Configuration for a single episode run."""

    stage: int = 2
    enabled_tiers: list[str] | None = None
    seeds: list[int] | None = None
    max_steps: int = 40
    synthesis_cadence: int = 8
    temperature: float = 0.7
    max_tokens: int = 512
    local_log_dir: str = "./episodes"
    stop_phrases: list[str] = field(
        default_factory=lambda: [
            "no more conjectures",
            "I have no further conjectures",
        ]
    )


@dataclass
class ServiceConfig:
    """Connection details for the MC-MCP service."""

    service_url: str = "ws://localhost:9090"
    api_key: str = ""
    session_create_url: str = ""

    def __post_init__(self) -> None:
        if not self.session_create_url:
            http_url = self.service_url.rstrip("/")
            http_url = http_url.replace("ws://", "http://", 1)
            http_url = http_url.replace("wss://", "https://", 1)
            self.session_create_url = f"{http_url}/v1/sessions"


@dataclass
class ModelConfig:
    """Model serving details."""

    backend: str = "vllm"
    model: str = "qwen3-8b"
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"


@dataclass
class ClientConfig:
    """Grouped runtime configuration for the client."""

    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: str = "mc_mcp_config.yaml") -> tuple[EpisodeConfig, ServiceConfig, ModelConfig]:
    """Load config from YAML and apply environment overrides."""

    raw_config = _load_yaml_mapping(Path(path))
    expanded_config = _expand_env_vars(raw_config)

    episode_data = _section_kwargs(EpisodeConfig, expanded_config.get("episode"))
    service_data = _section_kwargs(ServiceConfig, expanded_config.get("service"))
    model_data = _section_kwargs(ModelConfig, expanded_config.get("model"))

    api_key = os.getenv("MC_MCP_API_KEY")
    service_url = os.getenv("MC_MCP_SERVICE_URL")
    model_url = os.getenv("MC_MCP_MODEL_URL")

    if api_key is not None:
        service_data["api_key"] = api_key
    if service_url is not None:
        service_data["service_url"] = service_url
        if "session_create_url" not in service_data:
            service_data["session_create_url"] = ""
    if model_url is not None:
        model_data["base_url"] = model_url

    return EpisodeConfig(**episode_data), ServiceConfig(**service_data), ModelConfig(**model_data)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a top-level mapping.")
    return loaded


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


__all__ = [
    "ClientConfig",
    "EpisodeConfig",
    "ServiceConfig",
    "ModelConfig",
    "load_config",
]
