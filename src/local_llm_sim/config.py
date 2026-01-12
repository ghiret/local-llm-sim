"""Configuration management using pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Required
    openrouter_api_key: str = Field(..., validation_alias="OPENROUTER_API_KEY")

    # Server
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=11434, validation_alias="PORT")

    # Simulation
    simulate_latency: bool = Field(default=True, validation_alias="SIMULATE_LATENCY")

    # Paths
    config_path: Path = Field(default=Path("./config"), validation_alias="CONFIG_PATH")

    # Logging
    log_level: str = Field(default="info", validation_alias="LOG_LEVEL")

    # OpenRouter
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Global settings instance - lazily loaded to allow testing
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (for testing)."""
    global _settings
    _settings = None
