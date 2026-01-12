"""Format conversion between Ollama and OpenAI message formats."""

from datetime import UTC, datetime
from typing import Any

from ..models.ollama import OllamaMessage


def ollama_messages_to_openai(messages: list[OllamaMessage]) -> list[dict[str, Any]]:
    """
    Convert Ollama message format to OpenAI format.

    Handles:
    - Basic text messages
    - Image attachments (base64)
    - Tool calls
    """
    openai_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.images:
            # Multimodal message
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for image in msg.images:
                # Assume JPEG if not specified
                if not image.startswith("data:"):
                    image = f"data:image/jpeg;base64,{image}"
                content.append({"type": "image_url", "image_url": {"url": image}})
            openai_messages.append(
                {
                    "role": msg.role,
                    "content": content,
                }
            )
        elif msg.tool_calls:
            # Tool call message
            openai_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                }
            )
        else:
            # Simple text message
            openai_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

    return openai_messages


def openai_chunk_to_ollama(chunk: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert an OpenAI streaming chunk to Ollama format."""
    choices = chunk.get("choices", [{}])
    choice = choices[0] if choices else {}
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    message: dict[str, Any] = {
        "role": delta.get("role", "assistant"),
        "content": delta.get("content", "") or "",
    }

    if "tool_calls" in delta:
        message["tool_calls"] = delta["tool_calls"]

    response: dict[str, Any] = {
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
        "message": message,
        "done": finish_reason is not None,
    }

    if finish_reason:
        response["done_reason"] = finish_reason

    return response


def openai_response_to_ollama(response: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert a non-streaming OpenAI response to Ollama format."""
    choices = response.get("choices", [{}])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    usage = response.get("usage", {})

    ollama_message: dict[str, Any] = {
        "role": message.get("role", "assistant"),
        "content": message.get("content", "") or "",
    }

    tool_calls = message.get("tool_calls")
    if tool_calls:
        ollama_message["tool_calls"] = tool_calls

    return {
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
        "message": ollama_message,
        "done": True,
        "done_reason": choice.get("finish_reason", "stop"),
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }
