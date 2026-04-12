from __future__ import annotations

from pathlib import Path

import pytest

from mc_mcp_client.config import (
    ClientConfig,
    DEFAULT_ENABLED_TIERS,
    DEFAULT_SERVICE_URL,
    DEFAULT_SESSION_BUDGET,
    DEFAULT_SESSION_SYNTHESIS_CADENCE,
    EpisodeConfig,
    ModelConfig,
    ServiceConfig,
    SessionConfig,
    load_client_config,
    load_config,
)


def test_service_config_derives_session_create_url() -> None:
    config = ServiceConfig(service_url="wss://api.mc-mcp.com")

    assert config.session_create_url == "https://api.mc-mcp.com/v1/sessions"


def test_service_config_defaults_to_hosted_service() -> None:
    config = ServiceConfig()

    assert config.service_url == DEFAULT_SERVICE_URL
    assert config.session_create_url == "https://api.mc-mcp.com/v1/sessions"


def test_session_config_builds_public_create_payload_defaults() -> None:
    session = SessionConfig()

    assert session.to_create_payload() == {
        "enabled_tiers": list(DEFAULT_ENABLED_TIERS),
        "budget": DEFAULT_SESSION_BUDGET,
        "synthesis_cadence": DEFAULT_SESSION_SYNTHESIS_CADENCE,
    }


def test_session_config_builds_public_create_payload_with_explicit_override() -> None:
    session = SessionConfig(
        session_id="sess_123",
        enabled_tiers=["E0", "E3"],
        budget=12,
        synthesis_cadence=5,
        family_config={"mode": "negabase", "base": 2},
    )

    assert session.to_create_payload() == {
        "session_id": "sess_123",
        "enabled_tiers": ["E0", "E3"],
        "budget": 12,
        "synthesis_cadence": 5,
        "family_config": {"mode": "negabase", "base": 2},
    }


def test_load_config_returns_defaults_when_file_is_missing(tmp_path: Path) -> None:
    session, episode, service, model = load_config(str(tmp_path / "missing.yaml"))

    assert session == SessionConfig()
    assert episode == EpisodeConfig()
    assert service == ServiceConfig()
    assert model == ModelConfig()


def test_load_client_config_returns_grouped_defaults_when_file_is_missing(tmp_path: Path) -> None:
    client = load_client_config(str(tmp_path / "missing.yaml"))

    assert client == ClientConfig()


def test_load_config_reads_yaml_and_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MC_MCP_API_KEY", "yaml-env-key")
    config_path = tmp_path / "mc_mcp_config.yaml"
    config_path.write_text(
        """
service:
  service_url: "wss://api.mc-mcp.com"
  api_key: "${MC_MCP_API_KEY}"
model:
  backend: "vllm"
  model: "qwen3-8b"
  base_url: "http://localhost:8080/v1"
session:
  enabled_tiers: ["E0", "E1", "E2"]
  budget: 60
  synthesis_cadence: 10
episode:
  stage: 3
  seeds: [11, 13]
  max_steps: 60
  temperature: 0.2
  max_tokens: 1024
  local_log_dir: "./tmp-episodes"
""",
        encoding="utf-8",
    )

    session, episode, service, model = load_config(str(config_path))

    assert session.enabled_tiers == ["E0", "E1", "E2"]
    assert session.budget == 60
    assert session.synthesis_cadence == 10
    assert episode.stage == 3
    assert episode.seeds == [11, 13]
    assert episode.max_steps == 60
    assert episode.synthesis_cadence == EpisodeConfig().synthesis_cadence
    assert episode.temperature == 0.2
    assert episode.max_tokens == 1024
    assert episode.local_log_dir == "./tmp-episodes"
    assert episode.stop_phrases == EpisodeConfig().stop_phrases
    assert service.service_url == "wss://api.mc-mcp.com"
    assert service.api_key == "yaml-env-key"
    assert service.session_create_url == "https://api.mc-mcp.com/v1/sessions"
    assert model.backend == "vllm"
    assert model.model == "qwen3-8b"
    assert model.base_url == "http://localhost:8080/v1"


def test_environment_overrides_take_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "mc_mcp_config.yaml"
    config_path.write_text(
        """
service:
  service_url: "ws://yaml-service:9090"
  api_key: "yaml-key"
model:
  base_url: "http://yaml-model:8080/v1"
episode:
  stage: 5
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("MC_MCP_API_KEY", "env-key")
    monkeypatch.setenv("MC_MCP_SERVICE_URL", "wss://env-service.mc-mcp.com")
    monkeypatch.setenv("MC_MCP_MODEL_URL", "http://env-model:9000/v1")

    session, episode, service, model = load_config(str(config_path))

    assert session == SessionConfig()
    assert episode.stage == 5
    assert service.api_key == "env-key"
    assert service.service_url == "wss://env-service.mc-mcp.com"
    assert service.session_create_url == "https://env-service.mc-mcp.com/v1/sessions"
    assert model.base_url == "http://env-model:9000/v1"


def test_load_config_promotes_legacy_episode_enabled_tiers_to_session(tmp_path: Path) -> None:
    config_path = tmp_path / "mc_mcp_config.yaml"
    config_path.write_text(
        """
episode:
  enabled_tiers: ["E0", "E1"]
  max_steps: 12
  synthesis_cadence: 6
  stage: 1
""",
        encoding="utf-8",
    )

    session, episode, service, model = load_config(str(config_path))

    assert session.enabled_tiers == ["E0", "E1"]
    assert session.budget == 12
    assert session.synthesis_cadence == 6
    assert episode == EpisodeConfig(stage=1, max_steps=12, synthesis_cadence=6)
    assert service == ServiceConfig()
    assert model == ModelConfig()


def test_load_config_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    config_path = tmp_path / "mc_mcp_config.yaml"
    config_path.write_text("- invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_config(str(config_path))
