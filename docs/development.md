# Development Guide

This guide covers setting up and working with local-llm-sim for development.

## Prerequisites

- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://www.docker.com/) (or Podman)
- An [OpenRouter API key](https://openrouter.ai/keys)

## Development Environment

### Using DevContainers (Recommended)

The easiest way to get started is using VS Code Dev Containers:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/local-llm-sim.git
   cd local-llm-sim
   ```

2. Open in VS Code:
   ```bash
   code .
   ```

3. When prompted, click "Reopen in Container" (or use Command Palette: `Dev Containers: Reopen in Container`)

4. Wait for the container to build and dependencies to install

5. Set your API key:
   ```bash
   # Option 1: Export environment variable
   export OPENROUTER_API_KEY="sk-or-..."

   # Option 2: Create .env file
   cp .env.example .env
   # Edit .env with your key
   ```

6. Start the services:
   ```bash
   source .venv/bin/activate
   ./scripts/dev.sh start-all
   ```

### What's Included in DevContainer

The development container includes:

- Python 3.12 with virtual environment
- All project dependencies (production + dev)
- Open WebUI pre-installed via pipx
- Ruff for linting/formatting
- Pylance for type checking
- Git and GitHub CLI
- Ports 3000 (Open WebUI) and 11434 (API) forwarded

### Manual Setup

If you prefer not to use Dev Containers:

```bash
# Clone and enter directory
git clone https://github.com/yourusername/local-llm-sim.git
cd local-llm-sim

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install Open WebUI (optional)
pipx install open-webui
```

## Running the Services

### Development Script

The `scripts/dev.sh` helper script provides common commands:

```bash
./scripts/dev.sh start         # Start local-llm-sim only
./scripts/dev.sh webui         # Start Open WebUI only
./scripts/dev.sh start-all     # Start both services
./scripts/dev.sh install-webui # Install Open WebUI
./scripts/dev.sh test          # Run tests
./scripts/dev.sh curl-test     # Test API with curl
```

### Manual Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the simulator
local-llm-sim serve

# With auto-reload for development
local-llm-sim serve --reload

# Start Open WebUI (in another terminal)
open-webui serve --port 3000
```

### Access Points

- **local-llm-sim API**: http://localhost:11434
- **Open WebUI**: http://localhost:3000
- **Metrics**: http://localhost:11434/api/stats
- **Prometheus**: http://localhost:11434/metrics

## Project Structure

```
local-llm-sim/
├── src/local_llm_sim/     # Main package
│   ├── api/               # FastAPI route handlers
│   ├── core/              # Business logic
│   └── models/            # Pydantic models
├── config/                # Model configurations
├── tests/                 # Test suite
├── docs/                  # Documentation
└── scripts/               # Helper scripts
```

## Making Changes

### Code Style

- Python 3.11+ with type hints everywhere
- Use `async/await` for all I/O operations
- Pydantic models for all API contracts
- Line length: 100 characters
- Format with Ruff (auto-runs on save in VS Code)

### Running Tests

```bash
# All tests
pytest -v

# With coverage
pytest --cov=local_llm_sim --cov-report=html

# Specific test file
pytest tests/test_latency.py -v

# Specific test
pytest tests/test_latency.py::TestLatencySimulator::test_prefill_delay -v
```

### Type Checking

```bash
mypy src/local_llm_sim
```

### Linting

```bash
# Check
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format
ruff format .
```

## Adding a New Model

1. Edit `config/models.yaml`:
   ```yaml
   models:
     my-model:m3:
       backend: "provider/model-id"
       display_name: "My Model [M3 Ultra]"
       latency:
         prefill_tps: 500
         decode_tps: 30
       capabilities:
         tools: true
         vision: false
         embeddings: false

     my-model:api:
       backend: "provider/model-id"
       display_name: "My Model [API]"
       latency:
         prefill_tps: 999999
         decode_tps: 999999
       capabilities:
         tools: true
         vision: false
         embeddings: false
   ```

2. Restart the server to load the new model

3. Verify:
   ```bash
   curl http://localhost:11434/api/tags | jq '.models[].name'
   ```

## Adding a New API Endpoint

1. Add the route handler in `src/local_llm_sim/api/ollama.py`:
   ```python
   @router.get("/new-endpoint")
   async def new_endpoint(request: Request) -> dict:
       """Description of what this endpoint does."""
       return {"status": "ok"}
   ```

2. Add Pydantic models if needed in `src/local_llm_sim/models/`

3. Add tests in `tests/test_api.py`

## Modifying Latency Simulation

The latency model is in `src/local_llm_sim/core/latency.py`:

```python
@dataclass
class LatencyConfig:
    prefill_tps: float
    decode_tps: float

    # Adjust these parameters to change behavior
    prefill_jitter_std: float = 0.12      # Prefill variance
    decode_jitter_std: float = 0.20       # Decode variance
    stutter_probability: float = 0.05     # Chance of stutter
    thermal_decay_start: int = 200        # Thermal slowdown start
    thermal_decay_rate: float = 0.0003    # Slowdown rate
```

## Debugging

### Enable Debug Logging

```bash
LOG_LEVEL=debug local-llm-sim serve
```

### View Request Flow

The `/api/stats/requests` endpoint shows recent requests with timing details:

```bash
curl http://localhost:11434/api/stats/requests | jq
```

### Disable Latency Simulation

For faster iteration when testing non-latency features:

```bash
SIMULATE_LATENCY=false local-llm-sim serve
```

Or use the `:api` model variants which have no simulation.

## Documentation

### PlantUML Diagrams

Diagrams are in `docs/diagrams/`. To regenerate PNGs after editing:

```bash
# Install PlantUML if not present
sudo apt-get install plantuml

# Regenerate all diagrams
plantuml -tpng docs/diagrams/*.puml
```

### Updating Documentation

- `docs/architecture.md` - Design decisions and component overview
- `docs/implementation.md` - Code patterns and API reference
- `docs/development.md` - This file
- `README.md` - User-facing quick start

## Contributing

1. Create a feature branch
2. Make changes with tests
3. Run `ruff check` and `pytest`
4. Submit a pull request

## Troubleshooting

### "Model not found" errors

Check that the model name matches exactly what's in `config/models.yaml`. The `:m3` or `:api` suffix is required.

### Open WebUI not connecting

1. Ensure local-llm-sim is running on port 11434
2. In Open WebUI settings, set Ollama URL to `http://localhost:11434`

### OpenRouter API errors

1. Check your API key is set: `echo $OPENROUTER_API_KEY`
2. Verify the model backend exists on OpenRouter
3. Check OpenRouter status: https://status.openrouter.ai

### High latency even with `:api` models

The `:api` models should have near-zero simulated latency. If you're seeing high latency, it's likely network latency to OpenRouter, not simulation.
