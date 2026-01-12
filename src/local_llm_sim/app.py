"""FastAPI application factory."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import metrics, ollama
from .config import get_settings
from .core.proxy import OpenRouterProxy
from .core.registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan - startup and shutdown."""
    settings = get_settings()

    # Startup
    app.state.registry = ModelRegistry(settings.config_path)
    app.state.proxy = OpenRouterProxy(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    app.state.simulate_latency = settings.simulate_latency

    yield

    # Shutdown
    await app.state.proxy.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="local-llm-sim",
        description="Simulate local LLM inference latency using cloud APIs",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include routers
    app.include_router(ollama.router)
    app.include_router(metrics.router)

    @app.get("/")
    async def root() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "service": "local-llm-sim"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint for Docker."""
        return {"status": "healthy"}

    return app
