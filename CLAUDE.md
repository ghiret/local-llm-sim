# CLAUDE.md - AI Assistant Context

This file provides context for AI assistants (Claude Code, Cursor, etc.) working on this project.

## Project Overview

**local-llm-sim** is a proxy server that emulates the Ollama API while adding realistic latency simulation to cloud LLM API calls (via OpenRouter). The goal is to let users experience what local LLM inference feels like on specific hardware (Mac Studio M3 Ultra) before purchasing.

## Key Documents

- **README.md** — User-facing documentation, quick start, usage examples
- **docs/architecture.md** — Architecture decisions, component design, rationale
- **docs/implementation.md** — Detailed implementation specification with code examples
- **docs/development.md** — Development environment setup and contribution guide

**Always read docs/implementation.md before implementing any component** — it contains the exact Pydantic models, API endpoint implementations, and test patterns to follow.

## Architecture Summary

```
User Tools (aider, Open WebUI, agents)
           │
           ▼
    ┌─────────────────────┐
    │   local-llm-sim     │
    │   (FastAPI server)  │
    │                     │
    │  • Ollama API       │ ← Emulates Ollama endpoints
    │  • Latency Sim      │ ← Adds realistic delays
    │  • Model Registry   │ ← Maps models to backends
    │  • Metrics          │ ← Tracks performance stats
    └─────────────────────┘
           │
           ▼
       OpenRouter API
```

## Core Components

### 1. Latency Simulator (`src/local_llm_sim/core/latency.py`)
- Simulates prefill (prompt processing) and decode (generation) phases
- Includes stochastic variance: gaussian jitter, random stutters, thermal decay
- See docs/implementation.md `LatencySimulator` class for exact implementation

### 2. Model Registry (`src/local_llm_sim/core/registry.py`)
- Loads model configs from `config/models.yaml`
- Maps friendly names (e.g., `deepseek-v3:m3`) to OpenRouter model IDs
- Stores latency parameters per model
- Each model has `:m3` (simulated) and `:api` (passthrough) variants

### 3. OpenRouter Proxy (`src/local_llm_sim/core/proxy.py`)
- Handles actual API calls to OpenRouter
- Manages streaming responses
- Single httpx async client

### 4. Format Converter (`src/local_llm_sim/core/converter.py`)
- Translates between Ollama and OpenAI message formats
- Handles multimodal (images), tool calls

### 5. API Handlers (`src/local_llm_sim/api/`)
- `ollama.py` — /api/chat, /api/generate, /api/tags, etc.
- `metrics.py` — /api/stats, /metrics (Prometheus)
- `openai.py` — Optional /v1/chat/completions compatibility

## Implementation Order

When building features, follow this order:

1. Configuration loading (config.py)
2. Model registry (registry.py)
3. Latency simulator (latency.py)
4. OpenRouter proxy (proxy.py)
5. Format converter (converter.py)
6. Ollama API endpoints (api/ollama.py)
7. Metrics (api/metrics.py)
8. Tests

## Key Design Decisions

1. **Ollama API as primary interface** — maximum ecosystem compatibility
2. **OpenRouter only backend** — simplifies scope, covers most models
3. **Stochastic latency** — realistic feel, not just constant delays
4. **YAML config over Modelfiles** — clearer for proxy configuration
5. **Metrics built-in** — core value prop is understanding latency impact

## Latency Model Details

```python
# Prefill: time to first token
prefill_delay = (input_tokens / prefill_tps) × (1 + gaussian(0, 0.12))

# Decode: per-token delay
base_delay = 1 / decode_tps
jitter = gaussian(0, 0.20)
stutter = 2-4x multiplier with 5% probability
thermal = gradual slowdown after 200 tokens
token_delay = base_delay × (1 + jitter) × stutter × thermal
```

## Testing

- Use pytest with pytest-asyncio
- Mock OpenRouter calls in unit tests
- Test latency distributions statistically
- See docs/implementation.md for test examples

## Common Tasks

### Adding a new model
1. Add entry to `config/models.yaml`
2. Specify backend OpenRouter model ID
3. Set latency.prefill_tps and latency.decode_tps
4. List capabilities (tools, vision, embeddings)

### Adjusting latency parameters
- Modify `config/models.yaml`
- Or create new model entry with different speeds
- Use `:api` model variants for no-simulation baseline (e.g., `deepseek-v3:api`)

### Adding a new API endpoint
1. Add to appropriate router in `src/local_llm_sim/api/`
2. Define Pydantic models in `src/local_llm_sim/models/`
3. Add tests in `tests/`

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-...   # Required
HOST=0.0.0.0                    # Default
PORT=11434                      # Ollama default port
SIMULATE_LATENCY=true           # Enable/disable simulation
CONFIG_PATH=./config            # Config directory
LOG_LEVEL=info                  # Logging level
```

## Dependencies

Core:
- FastAPI + uvicorn (web server)
- httpx (async HTTP client)
- pydantic (data validation)
- tiktoken (token counting)
- PyYAML (config loading)
- click + rich (CLI)

Dev:
- pytest + pytest-asyncio
- ruff (linting)
- mypy (type checking)

## Code Style

- Python 3.11+ with type hints
- Async/await for all I/O operations
- Pydantic models for all API contracts
- 100 char line length (ruff)
- Follow existing patterns in SPEC.md

## Questions to Ask When Uncertain

1. Does this match the Ollama API behavior? (Check https://github.com/ollama/ollama/blob/main/docs/api.md)
2. Is this latency simulation realistic? (Check docs/architecture.md variance parameters)
3. Does the OpenRouter format conversion handle this case?
4. Are metrics being recorded for this operation?
