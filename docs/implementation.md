# Implementation Specification: local-llm-sim

This document provides detailed implementation specifications for local-llm-sim. It serves as a reference for the codebase structure and key implementation patterns.

## Project Structure

```
local-llm-sim/
├── README.md
├── CLAUDE.md              # AI assistant context
├── LICENSE
├── pyproject.toml
│
├── docs/
│   ├── architecture.md    # Design decisions and rationale
│   ├── implementation.md  # This file
│   ├── development.md     # Development setup guide
│   └── diagrams/
│       ├── architecture.puml
│       ├── request-flow.puml
│       └── latency-model.puml
│
├── .devcontainer/
│   ├── Dockerfile         # Dev environment container
│   └── devcontainer.json  # VS Code devcontainer config
│
├── src/
│   └── local_llm_sim/
│       ├── __init__.py
│       ├── main.py        # CLI entry point
│       ├── app.py         # FastAPI application factory
│       ├── config.py      # Configuration loading
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── ollama.py  # Ollama API endpoints
│       │   └── metrics.py # Stats and metrics endpoints
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── latency.py   # Latency simulation
│       │   ├── registry.py  # Model registry
│       │   ├── proxy.py     # OpenRouter proxy
│       │   └── converter.py # Format conversion
│       │
│       └── models/
│           ├── __init__.py
│           ├── ollama.py    # Ollama request/response models
│           └── config.py    # Configuration models
│
├── config/
│   └── models.yaml        # Model configurations
│
├── scripts/
│   └── dev.sh             # Development helper script
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_latency.py
    ├── test_registry.py
    ├── test_converter.py
    └── test_api.py
```

## Configuration

### Environment Variables

```python
# src/local_llm_sim/config.py
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # Required
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")

    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=11434, env="PORT")

    # Simulation
    simulate_latency: bool = Field(default=True, env="SIMULATE_LATENCY")

    # Paths
    config_path: Path = Field(default=Path("./config"), env="CONFIG_PATH")

    # Logging
    log_level: str = Field(default="info", env="LOG_LEVEL")

    # OpenRouter
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        env="OPENROUTER_BASE_URL"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
```

### Models Configuration

Models use a dual-variant naming convention:
- `:m3` - Mac Studio M3 Ultra simulation (with latency)
- `:api` - Direct API access (no latency, for benchmarking)

```yaml
# config/models.yaml

hardware:
  name: "Mac Studio M3 Ultra 512GB"
  description: "Apple Silicon with 512GB unified memory"

models:
  # Simulated version
  deepseek-v3:m3:
    backend: "deepseek/deepseek-chat-v3-0324"
    display_name: "DeepSeek V3 [M3 Ultra]"
    description: "671B MoE - Simulated on Mac Studio M3 Ultra"
    latency:
      prefill_tps: 400
      decode_tps: 20
    parameters:
      size: "671B"
      quantization: "Q4_K_M"
      context_length: 128000
    defaults:
      temperature: 0.7
      top_p: 0.9
    capabilities:
      tools: true
      vision: false
      embeddings: false
    size_bytes: 404000000000

  # Direct API version (no simulation)
  deepseek-v3:api:
    backend: "deepseek/deepseek-chat-v3-0324"
    display_name: "DeepSeek V3 [API]"
    description: "671B MoE - Direct API (no latency simulation)"
    latency:
      prefill_tps: 999999  # Effectively no delay
      decode_tps: 999999
    capabilities:
      tools: true
      vision: false
      embeddings: false
    size_bytes: 0
```

## Data Models

### Pydantic Models

```python
# src/local_llm_sim/models/config.py
from pydantic import BaseModel, Field

class LatencyConfig(BaseModel):
    prefill_tps: float = Field(..., description="Tokens per second for prefill")
    decode_tps: float = Field(..., description="Tokens per second for decode")

class CapabilitiesConfig(BaseModel):
    tools: bool = False
    vision: bool = False
    embeddings: bool = False

class ModelConfig(BaseModel):
    backend: str = Field(..., description="OpenRouter model identifier")
    display_name: str | None = None
    description: str | None = None
    latency: LatencyConfig
    parameters: dict = Field(default_factory=dict)
    defaults: dict = Field(default_factory=dict)
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)
    size_bytes: int = 0
```

```python
# src/local_llm_sim/models/ollama.py
from pydantic import BaseModel, Field
from typing import Literal

class OllamaMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    images: list[str] | None = None  # Base64 encoded
    tool_calls: list[dict] | None = None

class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = True
    format: str | None = None  # "json" for JSON mode
    options: dict | None = None
    tools: list[dict] | None = None
    keep_alive: str | None = None

class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
```

## Core Components

### Latency Simulator

```python
# src/local_llm_sim/core/latency.py
import asyncio
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, TypeVar

T = TypeVar("T")

@dataclass
class LatencyConfig:
    prefill_tps: float
    decode_tps: float

    # Variance parameters
    prefill_jitter_std: float = 0.12      # 12% standard deviation
    decode_jitter_std: float = 0.20       # 20% standard deviation
    stutter_probability: float = 0.05     # 5% chance per token
    stutter_multiplier_min: float = 2.0
    stutter_multiplier_max: float = 4.0
    thermal_decay_start: int = 200        # Start slowing after this many tokens
    thermal_decay_rate: float = 0.0003    # Slowdown per token after threshold

@dataclass
class LatencyStats:
    """Statistics from a single request's latency simulation."""
    input_tokens: int = 0
    output_tokens: int = 0
    prefill_delay_ms: float = 0
    decode_delay_ms: float = 0
    total_delay_ms: float = 0
    effective_prefill_tps: float = 0
    effective_decode_tps: float = 0
    stutters: int = 0

class LatencySimulator:
    """
    Simulates realistic local LLM inference latency.

    Models two phases:
    1. Prefill: Processing input tokens (time to first output token)
    2. Decode: Generating output tokens (streaming speed)

    Includes realistic variance:
    - Gaussian jitter on base rates
    - Random stutters (occasional longer delays)
    - Thermal throttling (gradual slowdown on long generations)
    """

    def __init__(self, config: LatencyConfig):
        self.config = config

    def _gaussian_factor(self, std: float) -> float:
        """Return a multiplier based on gaussian distribution."""
        return 1 + random.gauss(0, std)

    def _calculate_prefill_delay(self, input_tokens: int) -> float:
        """Calculate prefill delay in seconds with jitter."""
        if self.config.prefill_tps <= 0:
            return 0

        base_delay = input_tokens / self.config.prefill_tps
        jitter = self._gaussian_factor(self.config.prefill_jitter_std)
        return max(0, base_delay * jitter)

    def _calculate_token_delay(self, token_index: int) -> float:
        """Calculate delay for a single token with all variance factors."""
        if self.config.decode_tps <= 0:
            return 0

        base_delay = 1 / self.config.decode_tps

        # Gaussian jitter
        jitter = self._gaussian_factor(self.config.decode_jitter_std)

        # Stutter (occasional longer delay)
        if random.random() < self.config.stutter_probability:
            stutter = random.uniform(
                self.config.stutter_multiplier_min,
                self.config.stutter_multiplier_max
            )
        else:
            stutter = 1.0

        # Thermal decay (gradual slowdown after threshold)
        if token_index > self.config.thermal_decay_start:
            thermal = 1 + (token_index - self.config.thermal_decay_start) * self.config.thermal_decay_rate
        else:
            thermal = 1.0

        return max(0, base_delay * jitter * stutter * thermal)

    async def simulate_prefill(self, input_tokens: int) -> float:
        """Simulate prefill phase. Returns delay in seconds."""
        delay = self._calculate_prefill_delay(input_tokens)
        if delay > 0:
            await asyncio.sleep(delay)
        return delay

    async def throttle_stream(
        self,
        stream: AsyncIterator[T],
        input_tokens: int,
    ) -> AsyncIterator[tuple[T, LatencyStats]]:
        """
        Wrap a stream with latency simulation.
        Yields (item, stats) tuples with incremental statistics.
        """
        stats = LatencyStats(input_tokens=input_tokens)

        # Prefill phase
        prefill_start = time.monotonic()
        prefill_delay = await self.simulate_prefill(input_tokens)
        stats.prefill_delay_ms = prefill_delay * 1000

        # Decode phase
        decode_start = time.monotonic()
        token_index = 0

        async for item in stream:
            token_delay = self._calculate_token_delay(token_index)

            if token_delay > 0:
                await asyncio.sleep(token_delay)

            token_index += 1
            stats.output_tokens = token_index
            stats.decode_delay_ms = (time.monotonic() - decode_start) * 1000
            stats.total_delay_ms = stats.prefill_delay_ms + stats.decode_delay_ms

            yield item, stats
```

### Model Registry

```python
# src/local_llm_sim/core/registry.py
import hashlib
import yaml
from pathlib import Path
from datetime import datetime, timezone

from ..models.config import ModelConfig
from ..models.ollama import OllamaModelInfo

class ModelRegistry:
    """Manages available models and their configurations."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._models: dict[str, ModelConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load models from YAML configuration file."""
        models_file = self.config_path / "models.yaml"
        if not models_file.exists():
            return

        with open(models_file) as f:
            data = yaml.safe_load(f)

        if "models" not in data:
            return

        for name, config in data["models"].items():
            self._models[name] = ModelConfig(**config)

    def _generate_digest(self, name: str) -> str:
        """Generate a fake digest for a model."""
        return hashlib.sha256(name.encode()).hexdigest()[:64]

    def list_models(self) -> list[OllamaModelInfo]:
        """Return all registered models in Ollama format."""
        models = []
        for name, config in self._models.items():
            family = "unknown"
            if "/" in config.backend:
                family = config.backend.split("/")[0]

            models.append(OllamaModelInfo(
                name=name,
                model=name,
                modified_at=datetime.now(timezone.utc).isoformat(),
                size=config.size_bytes,
                digest=self._generate_digest(name),
                details={
                    "parent_model": "",
                    "format": "gguf",
                    "family": family,
                    "parameter_size": config.parameters.get("size", "unknown"),
                    "quantization_level": config.parameters.get("quantization", "unknown"),
                }
            ))
        return models

    def get_model(self, name: str) -> ModelConfig | None:
        """Get a model configuration by name."""
        if name in self._models:
            return self._models[name]

        # Try without tag (e.g., "deepseek-v3" -> "deepseek-v3:m3")
        for model_name in self._models:
            if model_name.startswith(name + ":"):
                return self._models[model_name]

        return None
```

### OpenRouter Proxy

```python
# src/local_llm_sim/core/proxy.py
import httpx
import json
from typing import AsyncIterator

from ..config import settings

class OpenRouterProxy:
    """Proxies requests to OpenRouter API."""

    def __init__(self):
        self.base_url = settings.openrouter_base_url
        self.api_key = settings.openrouter_api_key
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

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict]:
        """Stream chat completions from OpenRouter."""
        client = await self._get_client()

        payload = {
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
                if not line or not line.startswith("data: "):
                    continue

                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    yield chunk
                except json.JSONDecodeError:
                    continue
```

### Format Converter

```python
# src/local_llm_sim/core/converter.py
from ..models.ollama import OllamaMessage

def ollama_messages_to_openai(messages: list[OllamaMessage]) -> list[dict]:
    """Convert Ollama message format to OpenAI format."""
    openai_messages = []

    for msg in messages:
        if msg.images:
            # Multimodal message
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            for image in msg.images:
                if not image.startswith("data:"):
                    image = f"data:image/jpeg;base64,{image}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image}
                })
            openai_messages.append({"role": msg.role, "content": content})
        elif msg.tool_calls:
            openai_messages.append({
                "role": msg.role,
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            })
        else:
            openai_messages.append({"role": msg.role, "content": msg.content})

    return openai_messages

def openai_chunk_to_ollama(chunk: dict, model: str) -> dict:
    """Convert an OpenAI streaming chunk to Ollama format."""
    choice = chunk.get("choices", [{}])[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    message = {
        "role": delta.get("role", "assistant"),
        "content": delta.get("content", ""),
    }

    if "tool_calls" in delta:
        message["tool_calls"] = delta["tool_calls"]

    response = {
        "model": model,
        "created_at": chunk.get("created", ""),
        "message": message,
        "done": finish_reason is not None,
    }

    if finish_reason:
        response["done_reason"] = finish_reason

    return response
```

## API Endpoints

### Ollama API Handlers

Key endpoints in `src/local_llm_sim/api/ollama.py`:

```python
@router.post("/chat")
async def chat(request_obj: OllamaChatRequest, request: Request) -> Any:
    """Chat completion with streaming and latency simulation."""
    # ...

@router.get("/tags")
async def list_tags(request: Request) -> OllamaTagsResponse:
    """List available models."""
    # ...

@router.post("/show")
async def show_model(request_obj: OllamaShowRequest, request: Request) -> dict:
    """Get model details."""
    # ...

@router.post("/embed")
async def embed(request: Request) -> dict:
    """Generate embeddings (newer Ollama endpoint)."""
    # ...

@router.get("/version")
async def get_version() -> dict:
    """Get Ollama version (simulated)."""
    return {"version": "0.5.0"}
```

### Latency Stats in Responses

The `/api/chat` endpoint appends latency statistics to streaming responses:

```python
# After streaming completes, append stats
if simulate_latency and (prefill_delay_ms > 0 or decode_delay_ms > 0):
    total_sim_ms = prefill_delay_ms + decode_delay_ms
    stats_summary = (
        f"\n\n---\n"
        f"*M3 Ultra sim: {total_sim_ms/1000:.1f}s total "
        f"({prefill_delay_ms/1000:.1f}s prefill @ {prefill_tps} t/s, "
        f"{decode_delay_ms/1000:.1f}s decode @ {decode_tps} t/s) | "
        f"API: {backend_ms/1000:.1f}s*"
    )
    # Yield as additional chunk
```

### Metrics API

```python
# src/local_llm_sim/api/metrics.py

@router.get("/api/stats")
async def get_stats() -> StatsResponse:
    """Get session statistics."""
    return metrics.get_stats()

@router.get("/api/stats/requests")
async def get_requests(limit: int = 100) -> list[RequestRecord]:
    """Get recent request records."""
    return metrics.get_recent_requests(limit)

@router.post("/api/stats/reset")
async def reset_stats():
    """Reset all statistics."""
    metrics.reset()
    return {"status": "reset"}

@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus-format metrics."""
    # Returns text/plain with Prometheus metric format
```

## Application Factory

```python
# src/local_llm_sim/app.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path

from .api import ollama, metrics
from .core.registry import ModelRegistry
from .core.proxy import OpenRouterProxy
from .config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.registry = ModelRegistry(settings.config_path)
    app.state.proxy = OpenRouterProxy()
    app.state.simulate_latency = settings.simulate_latency

    yield

    # Shutdown
    await app.state.proxy.close()

def create_app() -> FastAPI:
    app = FastAPI(
        title="local-llm-sim",
        description="Simulate local LLM inference latency",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(ollama.router)
    app.include_router(metrics.router)

    @app.get("/")
    async def root():
        return {"status": "ok", "service": "local-llm-sim"}

    return app
```

## CLI Entry Point

```python
# src/local_llm_sim/main.py
import click
import uvicorn
from rich.console import Console

from .config import settings

console = Console()

@click.group()
def cli():
    """local-llm-sim: Simulate local LLM inference latency."""
    pass

@cli.command()
@click.option("--host", default=settings.host, help="Bind host")
@click.option("--port", default=settings.port, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the simulator server."""
    console.print("[bold green]Starting local-llm-sim[/]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Latency simulation: {'enabled' if settings.simulate_latency else 'disabled'}")
    console.print()
    console.print(f"[dim]Ollama API: http://{host}:{port}/api[/]")
    console.print(f"[dim]Metrics: http://{host}:{port}/api/stats[/]")

    uvicorn.run(
        "local_llm_sim.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level,
    )
```

## Testing

### Test Configuration

```python
# tests/conftest.py
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from pathlib import Path
import tempfile
import yaml

from local_llm_sim.app import create_app
from local_llm_sim.config import settings

@pytest.fixture
def temp_config():
    """Create temporary config directory with test models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir)

        models_config = {
            "models": {
                "test-model:m3": {
                    "backend": "openai/gpt-3.5-turbo",
                    "latency": {"prefill_tps": 1000, "decode_tps": 100},
                    "capabilities": {"tools": True, "vision": False, "embeddings": False},
                    "size_bytes": 1000000,
                }
            }
        }

        with open(config_path / "models.yaml", "w") as f:
            yaml.dump(models_config, f)

        yield config_path

@pytest_asyncio.fixture
async def client(temp_config):
    """Create test client with temporary config."""
    settings.config_path = temp_config
    settings.simulate_latency = False  # Disable for faster tests

    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
```

### Example Tests

```python
# tests/test_latency.py
import pytest
import statistics
from local_llm_sim.core.latency import LatencySimulator, LatencyConfig

class TestLatencySimulator:
    def test_prefill_delay_calculation(self):
        config = LatencyConfig(prefill_tps=400, decode_tps=20)
        simulator = LatencySimulator(config)

        # 4000 tokens at 400 tps = 10 seconds base
        delays = [simulator._calculate_prefill_delay(4000) for _ in range(100)]

        mean_delay = statistics.mean(delays)
        assert 8 < mean_delay < 12  # Within ~20% of expected 10s

    def test_decode_stutter_occurs(self):
        config = LatencyConfig(
            prefill_tps=400, decode_tps=20,
            stutter_probability=0.1,  # 10% for easier testing
        )
        simulator = LatencySimulator(config)

        delays = [simulator._calculate_token_delay(i) for i in range(1000)]
        base_delay = 1 / 20

        # Some delays should be significantly longer (stutters)
        stutters = [d for d in delays if d > base_delay * 1.5]
        assert len(stutters) > 50  # Expect ~100 stutters

# tests/test_api.py
@pytest.mark.asyncio
async def test_list_tags(client):
    response = await client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data

@pytest.mark.asyncio
async def test_version_endpoint(client):
    response = await client.get("/api/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
```

## Implementation Order

For building or extending the project:

1. **Configuration** - Settings, YAML loading
2. **Model registry** - Load models, list, show, CRUD
3. **Latency simulator** - Core delay logic with stochastic behavior
4. **OpenRouter proxy** - Streaming proxy
5. **Format converter** - Ollama <-> OpenAI translation
6. **Ollama API** - /api/tags, /api/show first, then /api/chat
7. **Metrics** - Stats collection and endpoints
8. **Tests** - Unit and integration tests
