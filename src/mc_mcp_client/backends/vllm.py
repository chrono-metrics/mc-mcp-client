"""vLLM/OpenAI-compatible backend."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib import error, request

from mc_mcp_client.backends.base import LLMBackend


class VLLMBackend(LLMBackend):
    """OpenAI-compatible backend (works with vLLM, TGI, Ollama, etc.).

    Public examples use `url="http://host:port"` and this backend appends `/v1`
    internally. `base_url="http://host:port/v1"` remains available as an
    explicit compatibility alias.
    """

    def __init__(
        self,
        model: str,
        url: str = "http://localhost:8080",
        *,
        base_url: str | None = None,
        api_key: str = "not-needed",
        timeout: float = 30.0,
    ) -> None:
        if base_url is not None and url != "http://localhost:8080":
            raise ValueError("Pass either url or base_url, not both.")

        configured_url = base_url or url
        self.model = model
        self.base_url = self._normalize_base_url(configured_url)
        self.url = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.api_key = api_key
        self.timeout = timeout
        self._client: Any | None = None

    async def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        """POST to /v1/chat/completions and return the assistant content."""
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop is not None:
            kwargs["stop"] = stop

        try:
            response = await client.chat.completions.create(**kwargs)
        except self._openai_timeout_error() as exc:
            raise TimeoutError("Timed out waiting for model response.") from exc
        except self._openai_connection_error() as exc:
            raise ConnectionError(f"Failed to reach the model server: {exc}") from exc

        if not response.choices:
            raise ValueError("Model response contained no choices.")

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model response contained no text content.")
        return content

    async def check_health(self) -> bool:
        """Probe /health, then fall back to /v1/models."""
        for url in self._healthcheck_urls():
            try:
                status = await asyncio.to_thread(self._fetch_status, url)
            except OSError:
                continue
            except TimeoutError:
                continue

            if status == 200:
                return True

        return False

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    "The optional 'openai' dependency is required for VLLMBackend. "
                    "Install mc-mcp-client[vllm]."
                ) from exc

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )

        return self._client

    def _healthcheck_urls(self) -> list[str]:
        if self.base_url.endswith("/v1"):
            health_url = f"{self.base_url[:-3]}/health"
        else:
            health_url = f"{self.base_url}/health"
        models_url = f"{self.base_url}/models"
        return [health_url, models_url]

    @staticmethod
    def _normalize_base_url(url: str) -> str:
        normalized = url.rstrip("/")
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"

    def _fetch_status(self, url: str) -> int:
        req = request.Request(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return response.status
        except error.HTTPError as exc:
            return exc.code
        except TimeoutError:
            raise
        except error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TimeoutError("Timed out waiting for health check.") from exc
            raise OSError(str(exc)) from exc

    def _openai_timeout_error(self) -> type[Exception]:
        from openai import APITimeoutError

        return APITimeoutError

    def _openai_connection_error(self) -> type[Exception]:
        from openai import APIConnectionError

        return APIConnectionError


__all__ = ["VLLMBackend"]
