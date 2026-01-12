"""Ollama-compatible API endpoints."""

import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import tiktoken
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.converter import (
    ollama_messages_to_openai,
    openai_chunk_to_ollama,
    openai_response_to_ollama,
)
from ..core.latency import LatencyConfig, LatencySimulator
from .metrics import metrics

if TYPE_CHECKING:
    from ..core.proxy import OpenRouterProxy
    from ..core.registry import ModelRegistry

from ..models.ollama import (
    OllamaChatRequest,
    OllamaDeleteRequest,
    OllamaEmbeddingsRequest,
    OllamaEmbeddingsResponse,
    OllamaGenerateRequest,
    OllamaMessage,
    OllamaPullRequest,
    OllamaRunningModel,
    OllamaShowRequest,
    OllamaTagsResponse,
)

router = APIRouter(prefix="/api", tags=["ollama"])


def get_registry(request: Request) -> "ModelRegistry":
    """Get the model registry from app state."""
    return request.app.state.registry


def get_proxy(request: Request) -> "OpenRouterProxy":
    """Get the OpenRouter proxy from app state."""
    return request.app.state.proxy


def get_simulate_latency(request: Request) -> bool:
    """Check if latency simulation is enabled."""
    return request.app.state.simulate_latency


def count_tokens(text: str) -> int:
    """Estimate token count using tiktoken."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough estimate
        return len(text) // 4


def count_message_tokens(messages: list[OllamaMessage]) -> int:
    """Count tokens in a list of messages."""
    total = 0
    for msg in messages:
        if msg.content:
            total += count_tokens(msg.content)
    return total


async def stream_chat_response(
    request_obj: OllamaChatRequest,
    registry: "ModelRegistry",
    proxy: "OpenRouterProxy",
    simulate_latency: bool,
) -> AsyncIterator[bytes]:
    """Generate streaming chat response with latency simulation."""
    model_config = registry.get_model(request_obj.model)
    if not model_config:
        raise HTTPException(404, f"Model '{request_obj.model}' not found")

    # Get the full model name for response
    model_name = registry.get_model_name(request_obj.model) or request_obj.model

    # Convert messages to OpenAI format
    openai_messages = ollama_messages_to_openai(request_obj.messages)

    # Count input tokens
    input_tokens = count_message_tokens(request_obj.messages)

    # Get options
    options = request_obj.options or {}
    temperature = options.get("temperature", model_config.defaults.get("temperature"))
    top_p = options.get("top_p", model_config.defaults.get("top_p"))

    # Track timing
    backend_start = time.monotonic()

    # Create the backend stream
    backend_stream = proxy.chat_completion_stream(
        model=model_config.backend,
        messages=openai_messages,
        tools=request_obj.tools,
        temperature=temperature,
        top_p=top_p,
    )

    output_tokens = 0
    prefill_delay_ms = 0.0
    decode_delay_ms = 0.0
    simulated_sleep_ms = 0.0

    if simulate_latency:
        # Create latency simulator
        latency_config = LatencyConfig(
            prefill_tps=model_config.latency.prefill_tps,
            decode_tps=model_config.latency.decode_tps,
        )
        simulator = LatencySimulator(latency_config)

        # Apply latency simulation
        async for chunk, stats in simulator.throttle_stream(backend_stream, input_tokens):
            ollama_chunk = openai_chunk_to_ollama(chunk, model_name)
            output_tokens = stats.output_tokens
            prefill_delay_ms = stats.prefill_delay_ms
            decode_delay_ms = stats.decode_delay_ms
            simulated_sleep_ms = stats.simulated_sleep_ms
            yield json.dumps(ollama_chunk).encode() + b"\n"
    else:
        async for chunk in backend_stream:
            ollama_chunk = openai_chunk_to_ollama(chunk, model_name)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                output_tokens += 1
            yield json.dumps(ollama_chunk).encode() + b"\n"

    total_elapsed_ms = (time.monotonic() - backend_start) * 1000
    # Actual API time = total elapsed - simulated sleep delays
    actual_api_ms = total_elapsed_ms - simulated_sleep_ms

    # Append latency stats summary to the response (visible in UI)
    if simulate_latency and (prefill_delay_ms > 0 or decode_delay_ms > 0):
        total_sim_ms = prefill_delay_ms + decode_delay_ms
        prefill_tps = model_config.latency.prefill_tps
        decode_tps = model_config.latency.decode_tps
        stats_summary = (
            f"\n\n---\n"
            f"*M3 Ultra sim: {total_sim_ms/1000:.1f}s total "
            f"({prefill_delay_ms/1000:.1f}s prefill @ {prefill_tps} t/s, "
            f"{decode_delay_ms/1000:.1f}s decode @ {decode_tps} t/s) | "
            f"API: {actual_api_ms/1000:.1f}s*"
        )
        stats_chunk = {
            "model": model_name,
            "created_at": datetime.now(UTC).isoformat(),
            "message": {"role": "assistant", "content": stats_summary},
            "done": False,
        }
        yield json.dumps(stats_chunk).encode() + b"\n"

    # Record metrics
    metrics.record_request(
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        simulated_prefill_ms=prefill_delay_ms,
        simulated_decode_ms=decode_delay_ms,
        backend_ms=actual_api_ms,
    )


@router.post("/chat")
async def chat(request_obj: OllamaChatRequest, request: Request) -> Any:
    """Chat completion endpoint."""
    registry = get_registry(request)
    proxy = get_proxy(request)
    simulate_latency = get_simulate_latency(request)

    if request_obj.stream:
        return StreamingResponse(
            stream_chat_response(request_obj, registry, proxy, simulate_latency),
            media_type="application/x-ndjson",
        )
    else:
        # Non-streaming
        model_config = registry.get_model(request_obj.model)
        if not model_config:
            raise HTTPException(404, f"Model '{request_obj.model}' not found")

        model_name = registry.get_model_name(request_obj.model) or request_obj.model
        openai_messages = ollama_messages_to_openai(request_obj.messages)
        input_tokens = count_message_tokens(request_obj.messages)

        backend_start = time.monotonic()
        prefill_delay_ms = 0.0
        simulated_sleep_ms = 0.0

        # Simulate prefill delay even for non-streaming
        if simulate_latency:
            latency_config = LatencyConfig(
                prefill_tps=model_config.latency.prefill_tps,
                decode_tps=model_config.latency.decode_tps,
            )
            simulator = LatencySimulator(latency_config)
            prefill_delay, prefill_sleep = await simulator.simulate_prefill(input_tokens)
            prefill_delay_ms = prefill_delay * 1000
            simulated_sleep_ms = prefill_sleep * 1000

        options = request_obj.options or {}
        response = await proxy.chat_completion(
            model=model_config.backend,
            messages=openai_messages,
            tools=request_obj.tools,
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
        )

        total_elapsed_ms = (time.monotonic() - backend_start) * 1000
        actual_api_ms = total_elapsed_ms - simulated_sleep_ms

        # Get output token count from response
        usage = response.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)

        # Record metrics
        metrics.record_request(
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            simulated_prefill_ms=prefill_delay_ms,
            simulated_decode_ms=0,
            backend_ms=actual_api_ms,
        )

        # Convert to Ollama format
        ollama_response = openai_response_to_ollama(response, model_name)

        # Append latency stats to non-streaming response
        if simulate_latency and prefill_delay_ms > 0:
            prefill_tps = model_config.latency.prefill_tps
            stats_summary = (
                f"\n\n---\n"
                f"*M3 Ultra sim: {prefill_delay_ms/1000:.1f}s prefill @ {prefill_tps} t/s | "
                f"API: {actual_api_ms/1000:.1f}s*"
            )
            if ollama_response.get("message", {}).get("content"):
                ollama_response["message"]["content"] += stats_summary

        return ollama_response


@router.post("/generate")
async def generate(request_obj: OllamaGenerateRequest, request: Request) -> Any:
    """Text generation endpoint (converts to chat internally)."""
    messages: list[OllamaMessage] = []
    if request_obj.system:
        messages.append(OllamaMessage(role="system", content=request_obj.system))
    messages.append(OllamaMessage(role="user", content=request_obj.prompt))

    # Convert to chat request
    chat_request = OllamaChatRequest(
        model=request_obj.model,
        messages=messages,
        stream=request_obj.stream,
        options=request_obj.options,
        format=request_obj.format,
    )

    return await chat(chat_request, request)


@router.get("/tags")
async def list_tags(request: Request) -> OllamaTagsResponse:
    """List available models."""
    registry = get_registry(request)
    return OllamaTagsResponse(models=registry.list_models())


@router.post("/show")
async def show_model(request_obj: OllamaShowRequest, request: Request) -> dict[str, Any]:
    """Get model details."""
    registry = get_registry(request)
    model_config = registry.get_model(request_obj.name)
    if not model_config:
        raise HTTPException(404, f"Model '{request_obj.name}' not found")

    # Determine family from backend
    family = "unknown"
    if "/" in model_config.backend:
        family = model_config.backend.split("/")[0]

    return {
        "modelfile": f"FROM {model_config.backend}",
        "parameters": model_config.parameters,
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "parameter_size": model_config.parameters.get("size", "unknown"),
            "quantization_level": model_config.parameters.get("quantization", "unknown"),
        },
        "model_info": {
            "backend": model_config.backend,
            "latency": {
                "prefill_tps": model_config.latency.prefill_tps,
                "decode_tps": model_config.latency.decode_tps,
            },
            "capabilities": model_config.capabilities.model_dump(),
        },
    }


@router.post("/pull")
async def pull_model(request_obj: OllamaPullRequest, request: Request) -> Any:
    """
    'Pull' a model (validates it exists in config).

    In a real Ollama, this downloads the model.
    For us, it just validates the model is registered.
    """
    registry = get_registry(request)
    if not registry.model_exists(request_obj.name):
        raise HTTPException(404, f"Model '{request_obj.name}' not found in registry")

    # Return success status
    if request_obj.stream:

        async def stream_status() -> AsyncIterator[bytes]:
            yield json.dumps({"status": f"pulling {request_obj.name}"}).encode() + b"\n"
            yield json.dumps({"status": "success"}).encode() + b"\n"

        return StreamingResponse(stream_status(), media_type="application/x-ndjson")
    else:
        return {"status": "success"}


@router.delete("/delete")
async def delete_model(request_obj: OllamaDeleteRequest, request: Request) -> dict[str, str]:
    """Remove a model from the registry."""
    registry = get_registry(request)
    if registry.remove_model(request_obj.name):
        return {"status": "success"}
    else:
        raise HTTPException(404, f"Model '{request_obj.name}' not found")


@router.post("/embeddings")
async def embeddings(
    request_obj: OllamaEmbeddingsRequest, request: Request
) -> OllamaEmbeddingsResponse:
    """Generate embeddings (legacy endpoint)."""
    registry = get_registry(request)
    proxy = get_proxy(request)

    model_config = registry.get_model(request_obj.model)
    if not model_config:
        raise HTTPException(404, f"Model '{request_obj.model}' not found")

    if not model_config.capabilities.embeddings:
        raise HTTPException(400, f"Model '{request_obj.model}' does not support embeddings")

    response = await proxy.embeddings(
        model=model_config.backend,
        input_text=request_obj.prompt,
    )

    # Extract embeddings from OpenAI format
    data = response.get("data", [])
    if len(data) == 1:
        return OllamaEmbeddingsResponse(embedding=data[0]["embedding"])
    else:
        return OllamaEmbeddingsResponse(embedding=[d["embedding"] for d in data])


@router.post("/embed")
async def embed(request: Request) -> dict[str, Any]:
    """Generate embeddings (newer Ollama endpoint with input field)."""
    registry = get_registry(request)
    proxy = get_proxy(request)

    # Parse the request body manually to handle both 'input' and 'prompt'
    body = await request.json()
    model_name = body.get("model", "")
    input_data = body.get("input", body.get("prompt", ""))

    model_config = registry.get_model(model_name)
    if not model_config:
        raise HTTPException(404, f"Model '{model_name}' not found")

    if not model_config.capabilities.embeddings:
        raise HTTPException(400, f"Model '{model_name}' does not support embeddings")

    # Handle both single string and list of strings
    if isinstance(input_data, str):
        inputs = [input_data]
    else:
        inputs = input_data

    # Generate embeddings for each input
    all_embeddings = []
    for text in inputs:
        response = await proxy.embeddings(
            model=model_config.backend,
            input_text=text,
        )
        data = response.get("data", [])
        if data:
            all_embeddings.append(data[0]["embedding"])

    return {
        "model": model_name,
        "embeddings": all_embeddings,
    }


@router.get("/ps")
async def list_running(request: Request) -> dict[str, list[OllamaRunningModel]]:
    """List 'running' models (always returns all models as loaded)."""
    registry = get_registry(request)
    models = registry.list_models()
    return {
        "models": [
            OllamaRunningModel(
                name=m.name,
                model=m.model,
                size=m.size,
                digest=m.digest,
                expires_at=datetime.now(UTC).isoformat(),
                size_vram=m.size,
            )
            for m in models
        ]
    }


@router.get("/version")
async def get_version() -> dict[str, str]:
    """Get Ollama version (simulated)."""
    return {"version": "0.5.0"}
