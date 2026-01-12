"""Configuration models for model definitions."""

from pydantic import BaseModel, Field


class LatencyConfig(BaseModel):
    """Latency parameters for a model."""

    prefill_tps: float = Field(..., description="Tokens per second for prefill (prompt processing)")
    decode_tps: float = Field(..., description="Tokens per second for decode (generation)")


class CapabilitiesConfig(BaseModel):
    """Model capabilities."""

    tools: bool = False
    vision: bool = False
    embeddings: bool = False


class ModelConfig(BaseModel):
    """Complete model configuration."""

    backend: str = Field(..., description="OpenRouter model identifier")
    display_name: str | None = None
    description: str | None = None
    latency: LatencyConfig
    parameters: dict = Field(default_factory=dict)
    defaults: dict = Field(default_factory=dict)
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)
    size_bytes: int = 0
