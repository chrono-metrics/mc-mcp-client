"""Backend ABC (CLIENT-05)."""
from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response given a prompt."""
        # TODO: implement in CLIENT-05
