"""OpenRouter API proxy for making LLM requests."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OpenRouterProxy:
    """
    Proxies requests to OpenRouter API.

    Handles authentication and streaming responses.
    """

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/local-llm-sim",
                    "X-Title": "local-llm-sim",
                },
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream chat completions from OpenRouter.

        Yields parsed SSE data chunks.
        """
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue

                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    yield chunk
                except json.JSONDecodeError:
                    continue

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion."""
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def embeddings(
        self,
        model: str,
        input_text: str | list[str],
    ) -> dict[str, Any]:
        """Generate embeddings."""
        client = await self._get_client()

        payload = {
            "model": model,
            "input": input_text,
        }

        response = await client.post("/embeddings", json=payload)
        response.raise_for_status()
        return response.json()
