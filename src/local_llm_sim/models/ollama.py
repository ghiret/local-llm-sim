"""Ollama API request and response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class OllamaMessage(BaseModel):
    """A message in Ollama format."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    images: list[str] | None = None  # Base64 encoded
    tool_calls: list[dict[str, Any]] | None = None


class OllamaChatRequest(BaseModel):
    """Request body for /api/chat."""

    model: str
    messages: list[OllamaMessage]
    stream: bool = True
    format: str | None = None  # "json" for JSON mode
    options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    keep_alive: str | None = None


class OllamaChatResponse(BaseModel):
    """Response body for /api/chat (streaming chunk or final response)."""

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


class OllamaGenerateRequest(BaseModel):
    """Request body for /api/generate."""

    model: str
    prompt: str
    stream: bool = True
    system: str | None = None
    template: str | None = None
    context: list[int] | None = None
    options: dict[str, Any] | None = None
    format: str | None = None
    raw: bool = False
    keep_alive: str | None = None


class OllamaModelDetails(BaseModel):
    """Model details in Ollama format."""

    parent_model: str = ""
    format: str = "gguf"
    family: str = "unknown"
    parameter_size: str = "unknown"
    quantization_level: str = "unknown"


class OllamaModelInfo(BaseModel):
    """Model information returned by /api/tags."""

    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: OllamaModelDetails = Field(default_factory=OllamaModelDetails)


class OllamaTagsResponse(BaseModel):
    """Response body for /api/tags."""

    models: list[OllamaModelInfo]


class OllamaShowRequest(BaseModel):
    """Request body for /api/show."""

    name: str


class OllamaPullRequest(BaseModel):
    """Request body for /api/pull."""

    name: str
    insecure: bool = False
    stream: bool = True


class OllamaDeleteRequest(BaseModel):
    """Request body for /api/delete."""

    name: str


class OllamaEmbeddingsRequest(BaseModel):
    """Request body for /api/embeddings."""

    model: str
    prompt: str | list[str]
    options: dict[str, Any] | None = None
    keep_alive: str | None = None


class OllamaEmbeddingsResponse(BaseModel):
    """Response body for /api/embeddings."""

    embedding: list[float] | list[list[float]]


class OllamaRunningModel(BaseModel):
    """A running model in Ollama format."""

    name: str
    model: str
    size: int
    digest: str
    expires_at: str
    size_vram: int


class OllamaPsResponse(BaseModel):
    """Response body for /api/ps."""

    models: list[OllamaRunningModel]
