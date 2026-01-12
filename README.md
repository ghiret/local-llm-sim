# local-llm-sim

**Simulate local LLM inference latency using cloud APIs.**

Test what running large language models on a Mac Studio M3 Ultra (or other hardware) actually *feels* like—before spending £7,000+ on hardware.

## The Problem

You're considering buying a Mac Studio M3 Ultra with 512GB unified memory to run models like DeepSeek V3 (671B) locally. The specs say you'll get ~20 tokens/second generation and ~400 tokens/second prompt processing. But what does that *feel like* when you're actually using it?

- How annoying is a 20-second wait for the first token on a long prompt?
- Is 20 tok/s fast enough for interactive coding with aider/opencode?
- Will your agent workflows be frustratingly slow?

This tool lets you find out by simulating realistic local inference latency while using cloud APIs (OpenRouter) for the actual model responses.

> **Note on Latency Numbers**
>
> The latency parameters in this project are *estimates* based on community benchmarks and theoretical calculations. They haven't been validated on actual M3 Ultra hardware. Real performance varies with quantization, context length, thermals, and software versions. Treat these as approximate guidelines. See [docs/architecture.md](docs/architecture.md) for details.

## How It Works

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  Open WebUI     │     │                      │     │             │
│  aider          │────▶│   local-llm-sim      │────▶│  OpenRouter │
│  opencode       │◀────│   (Ollama API)       │◀────│             │
│  agents         │     │                      │     │             │
└─────────────────┘     └──────────────────────┘     └─────────────┘
                               │
                        Adds realistic delays:
                        • Prefill: ~400 tok/s
                        • Decode: ~20 tok/s
                        • Stochastic jitter
                        • Thermal throttling
```

The simulator exposes an **Ollama-compatible API**, so any tool that works with Ollama works with this—unchanged.

## Quick Start

### Prerequisites

- Python 3.11+
- An [OpenRouter API key](https://openrouter.ai/keys)

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/local-llm-sim.git
cd local-llm-sim

# Install
pip install -e .

# Set your API key
export OPENROUTER_API_KEY="sk-or-..."

# Run the simulator
local-llm-sim serve
```

The simulator runs on `http://localhost:11434`—the same port as Ollama.

### With Docker (includes Open WebUI)

```bash
# Set your API key
export OPENROUTER_API_KEY="sk-or-..."

# Start everything
docker compose up -d

# Open WebUI at http://localhost:3000
# Ollama API at http://localhost:11434
```

## Usage

### With Open WebUI

Just point Open WebUI at the simulator—it's already configured in docker-compose.yml.

### With aider

```bash
export OLLAMA_HOST=http://localhost:11434
aider --model ollama/deepseek-v3:m3
```

### With opencode

```bash
export OLLAMA_HOST=http://localhost:11434
opencode
```

### With any OpenAI-compatible client

The simulator also exposes `/v1/chat/completions`:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="deepseek-v3:m3",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

## Available Models

Each model has two versions for benchmarking:
- `:m3` - M3 Ultra simulation (realistic latency)
- `:api` - Direct API access (no latency, for comparison)

| Model | Backend | Prefill | Decode | Use Case |
|-------|---------|---------|--------|----------|
| `deepseek-v3:m3` | DeepSeek V3 | 400 t/s | 20 t/s | Best overall |
| `deepseek-r1:m3` | DeepSeek R1 | 350 t/s | 18 t/s | Reasoning |
| `llama4-scout:m3` | Llama 4 Scout (109B) | 800 t/s | 40 t/s | Fast assistant |
| `llama4-maverick:m3` | Llama 4 Maverick (400B) | 500 t/s | 25 t/s | Capable |
| `qwen2.5-72b:m3` | Qwen 2.5 72B | 600 t/s | 45 t/s | Fast |
| `nomic-embed:latest` | Nomic Embed v1.5 | — | — | Embeddings/RAG |

All models except embeddings also have `:api` versions (e.g., `deepseek-v3:api`).

List available models:
```bash
curl http://localhost:11434/api/tags
```

### Adding Custom Models

Create or edit `config/models.yaml`:

```yaml
models:
  my-custom-model:q4:
    backend: openrouter/anthropic/claude-3-haiku
    latency:
      prefill_tps: 1000
      decode_tps: 80
    capabilities:
      tools: true
      vision: false
```

Then pull it:
```bash
curl -X POST http://localhost:11434/api/pull -d '{"name": "my-custom-model:q4"}'
```

## Latency Simulation

The simulator models realistic local inference behavior:

### Prefill (Prompt Processing)
- Base rate from hardware profile (e.g., 400 tok/s for DeepSeek V3 on M3 Ultra)
- Gaussian jitter (σ = 12%)
- Longer prompts = longer wait for first token

### Decode (Token Generation)
- Base rate from hardware profile (e.g., 20 tok/s)
- Per-token gaussian jitter (σ = 20%)
- 5% chance of "stutter" (2-4x delay on a token)
- Thermal decay after 200+ tokens (gradual 0.03%/token slowdown)
- Bursty emission (1-3 tokens, then pause)

### Example Timeline

For a 4,000 token prompt generating 200 tokens at DeepSeek V3 speeds:

```
Prefill:  4000 / 400 = 10 seconds (±1.2s)
Decode:   200 / 20  = 10 seconds (±2s, with occasional stutters)
Total:    ~20 seconds

With cloud API (no simulation): ~2 seconds
```

## Metrics & Comparison

See how the simulation compares to direct API access:

```bash
curl http://localhost:11434/api/stats
```

```json
{
  "session": {
    "started": "2026-01-11T10:30:00Z",
    "requests": 47,
    "input_tokens": 89432,
    "output_tokens": 12847,
    "openrouter_cost_usd": 0.0234
  },
  "latency_comparison": {
    "simulated_total_sec": 847,
    "backend_total_sec": 52,
    "slowdown_factor": 16.3
  }
}
```

Prometheus metrics at `/metrics`:

```bash
curl http://localhost:11434/metrics
```

## Disabling Simulation

To compare with and without latency simulation:

```bash
# Use the passthrough model (no delays)
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-v3:api",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

Or set an environment variable:

```bash
SIMULATE_LATENCY=false local-llm-sim serve
```

## Tool Calling

Tool calling works with supported models:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-v3:m3",
  "messages": [{"role": "user", "content": "What is the weather in London?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"}
        },
        "required": ["location"]
      }
    }
  }]
}'
```

## Embeddings (for RAG)

```bash
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed:latest",
  "prompt": "Hello world"
}'
```

Works with Open WebUI's RAG features, LangChain, LlamaIndex, etc.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | Required. Your OpenRouter API key |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `11434` | Port (Ollama default) |
| `SIMULATE_LATENCY` | `true` | Enable/disable latency simulation |
| `CONFIG_PATH` | `./config` | Path to config directory |
| `LOG_LEVEL` | `info` | Logging level |

### Config Files

- `config/models.yaml` — Model definitions
- `config/hardware.yaml` — Hardware profile (M3 Ultra specs)

## Limitations

- **Not real inference** — Model quality comes from cloud APIs, not local compute
- **No actual memory constraints** — Won't simulate OOM or context limits
- **Network dependency** — Requires internet for OpenRouter
- **Approximation** — Real hardware has additional variance we don't model

## Why Not Just Use Ollama?

You could install Ollama and download models, but:

1. **You don't have the hardware yet** — That's the point of this tool
2. **Smaller models aren't representative** — Running 7B locally doesn't tell you how 671B feels
3. **Model quality differs** — This gives you the real DeepSeek V3/R1 responses at simulated local speeds

## Development

### Using the DevContainer (Recommended)

1. Open in VS Code with Dev Containers extension
2. "Reopen in Container" when prompted
3. The virtual environment is auto-created with all dependencies

```bash
# Activate the venv
source .venv/bin/activate

# Set your API key
export OPENROUTER_API_KEY="sk-or-..."

# Start local-llm-sim server
local-llm-sim serve

# In another terminal, start Open WebUI (auto-installs if needed)
./scripts/dev.sh webui

# Or start both together
./scripts/dev.sh start-all
```

### Dev Script Commands

```bash
./scripts/dev.sh start         # Start local-llm-sim only
./scripts/dev.sh webui         # Start Open WebUI only
./scripts/dev.sh start-all     # Start both services
./scripts/dev.sh install-webui # Install Open WebUI
./scripts/dev.sh test          # Run tests
./scripts/dev.sh curl-test     # Test API with curl
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Run with auto-reload
local-llm-sim serve --reload
```

## Contributing

Contributions welcome! See the documentation:
- [Architecture](docs/architecture.md) - Design decisions and component overview
- [Implementation](docs/implementation.md) - Code patterns and API reference
- [Development](docs/development.md) - Setup guide for contributors

## License

MIT
