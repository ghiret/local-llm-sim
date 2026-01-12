"""CLI entry point for local-llm-sim."""

import click
import uvicorn
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli() -> None:
    """local-llm-sim: Simulate local LLM inference latency."""
    pass


@cli.command()
@click.option("--host", default=None, help="Bind host (default: from config)")
@click.option("--port", default=None, type=int, help="Bind port (default: 11434)")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Start the simulator server."""
    from .config import get_settings

    settings = get_settings()

    # Use CLI args or fall back to settings
    bind_host = host or settings.host
    bind_port = port or settings.port

    console.print("[bold green]Starting local-llm-sim[/]")
    console.print(f"  Host: {bind_host}")
    console.print(f"  Port: {bind_port}")
    console.print(
        f"  Latency simulation: {'[green]enabled[/]' if settings.simulate_latency else '[yellow]disabled[/]'}"
    )
    console.print()
    console.print(f"[dim]Ollama API: http://{bind_host}:{bind_port}/api[/]")
    console.print(f"[dim]Metrics: http://{bind_host}:{bind_port}/api/stats[/]")
    console.print()

    uvicorn.run(
        "local_llm_sim.app:create_app",
        factory=True,
        host=bind_host,
        port=bind_port,
        reload=reload,
        log_level=settings.log_level,
    )


@cli.command("list-models")
def list_models() -> None:
    """List available models."""
    from .config import get_settings
    from .core.registry import ModelRegistry

    settings = get_settings()
    registry = ModelRegistry(settings.config_path)
    models = registry.list_models()

    if not models:
        console.print("[yellow]No models configured.[/]")
        console.print("Add models to config/models.yaml")
        return

    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Size")
    table.add_column("Prefill (t/s)", justify="right")
    table.add_column("Decode (t/s)", justify="right")
    table.add_column("Capabilities")

    for m in models:
        model_config = registry.get_model(m.name)
        if model_config:
            caps = []
            if model_config.capabilities.tools:
                caps.append("tools")
            if model_config.capabilities.vision:
                caps.append("vision")
            if model_config.capabilities.embeddings:
                caps.append("embeddings")

            table.add_row(
                m.name,
                m.details.parameter_size,
                str(int(model_config.latency.prefill_tps)),
                str(int(model_config.latency.decode_tps)),
                ", ".join(caps) if caps else "-",
            )

    console.print(table)


@cli.command("version")
def version() -> None:
    """Show version information."""
    console.print("[bold]local-llm-sim[/] version 0.1.0")


if __name__ == "__main__":
    cli()
