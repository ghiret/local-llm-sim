"""Test configuration and fixtures."""

import os
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
import yaml
from httpx import ASGITransport, AsyncClient

# Set a dummy API key for testing before importing the app
os.environ.setdefault("OPENROUTER_API_KEY", "test-api-key")


@pytest.fixture
def temp_config() -> Path:
    """Create temporary config directory with test models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir)

        models_config = {
            "hardware": {
                "name": "Test Hardware",
                "description": "Test hardware profile",
            },
            "models": {
                "test-model:latest": {
                    "backend": "openai/gpt-3.5-turbo",
                    "display_name": "Test Model",
                    "description": "A test model",
                    "latency": {
                        "prefill_tps": 1000,
                        "decode_tps": 100,
                    },
                    "parameters": {
                        "size": "7B",
                        "quantization": "Q4_K_M",
                    },
                    "capabilities": {
                        "tools": True,
                        "vision": False,
                        "embeddings": False,
                    },
                    "size_bytes": 1000000,
                },
                "embed-model:latest": {
                    "backend": "nomic-ai/nomic-embed-text-v1.5",
                    "display_name": "Test Embed Model",
                    "latency": {
                        "prefill_tps": 10000,
                        "decode_tps": 0,
                    },
                    "capabilities": {
                        "tools": False,
                        "vision": False,
                        "embeddings": True,
                    },
                    "size_bytes": 500000,
                },
            },
        }

        with open(config_path / "models.yaml", "w") as f:
            yaml.dump(models_config, f)

        yield config_path


@pytest_asyncio.fixture
async def client(temp_config: Path) -> AsyncGenerator[AsyncClient, None]:
    """Create test client with temporary config."""
    from local_llm_sim.config import reset_settings
    from local_llm_sim.core.proxy import OpenRouterProxy
    from local_llm_sim.core.registry import ModelRegistry

    # Reset settings to pick up test config
    reset_settings()

    # Update environment for test
    os.environ["CONFIG_PATH"] = str(temp_config)
    os.environ["SIMULATE_LATENCY"] = "false"  # Disable for faster tests

    # Import after setting environment variables
    from local_llm_sim.app import create_app

    app = create_app()

    # Manually set up app state since ASGITransport doesn't trigger lifespan
    app.state.registry = ModelRegistry(temp_config)
    app.state.proxy = OpenRouterProxy(
        api_key="test-api-key",
        base_url="https://openrouter.ai/api/v1",
    )
    app.state.simulate_latency = False

    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Clean up
    await app.state.proxy.close()
    reset_settings()
