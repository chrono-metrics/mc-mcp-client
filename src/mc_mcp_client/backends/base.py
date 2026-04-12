"""Backend interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Abstract interface for model inference."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        """Send a conversation and return the model's response text."""

    @abstractmethod
    async def check_health(self) -> bool:
        """Verify the model server is reachable and responding."""


# Backward-compatible alias while the rest of the package migrates to LLMBackend.
BaseBackend = LLMBackend

__all__ = ["LLMBackend", "BaseBackend"]
