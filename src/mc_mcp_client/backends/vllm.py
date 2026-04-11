"""vLLM/OpenAI-compatible backend (CLIENT-05)."""
from mc_mcp_client.backends.base import BaseBackend


class VLLMBackend(BaseBackend):
    """Backend that calls a vLLM server via the OpenAI-compatible API."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        # TODO: implement in CLIENT-05

    def generate(self, prompt: str, **kwargs) -> str:
        # TODO: implement in CLIENT-05
        raise NotImplementedError
