"""Model registry for managing available models."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..models.config import CapabilitiesConfig, LatencyConfig, ModelConfig
from ..models.ollama import OllamaModelDetails, OllamaModelInfo


class ModelRegistry:
    """
    Manages available models and their configurations.

    Models are loaded from YAML config and can be added/removed at runtime.
    """

    def __init__(self, config_path: Path) -> None:
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

        if not data or "models" not in data:
            return

        for name, config in data["models"].items():
            # Parse nested configs
            latency_data = config.get("latency", {})
            latency = LatencyConfig(
                prefill_tps=latency_data.get("prefill_tps", 100),
                decode_tps=latency_data.get("decode_tps", 20),
            )

            capabilities_data = config.get("capabilities", {})
            capabilities = CapabilitiesConfig(
                tools=capabilities_data.get("tools", False),
                vision=capabilities_data.get("vision", False),
                embeddings=capabilities_data.get("embeddings", False),
            )

            self._models[name] = ModelConfig(
                backend=config["backend"],
                display_name=config.get("display_name"),
                description=config.get("description"),
                latency=latency,
                parameters=config.get("parameters", {}),
                defaults=config.get("defaults", {}),
                capabilities=capabilities,
                size_bytes=config.get("size_bytes", 0),
            )

    def _generate_digest(self, name: str) -> str:
        """Generate a fake digest for a model."""
        return hashlib.sha256(name.encode()).hexdigest()[:64]

    def list_models(self) -> list[OllamaModelInfo]:
        """Return all registered models in Ollama format."""
        models = []
        for name, config in self._models.items():
            # Determine family from backend
            family = "unknown"
            if "/" in config.backend:
                family = config.backend.split("/")[0]

            details = OllamaModelDetails(
                parent_model="",
                format="gguf",
                family=family,
                parameter_size=config.parameters.get("size", "unknown"),
                quantization_level=config.parameters.get("quantization", "unknown"),
            )

            models.append(
                OllamaModelInfo(
                    name=name,
                    model=name,
                    modified_at=datetime.now(UTC).isoformat(),
                    size=config.size_bytes,
                    digest=self._generate_digest(name),
                    details=details,
                )
            )
        return models

    def get_model(self, name: str) -> ModelConfig | None:
        """Get a model configuration by name."""
        # Try exact match first
        if name in self._models:
            return self._models[name]

        # Try without tag (e.g., "deepseek-v3" -> "deepseek-v3:q4")
        for model_name in self._models:
            if model_name.startswith(name + ":") or model_name == name:
                return self._models[model_name]

        return None

    def get_model_name(self, name: str) -> str | None:
        """Get the full model name (for partial matches)."""
        if name in self._models:
            return name

        for model_name in self._models:
            if model_name.startswith(name + ":"):
                return model_name

        return None

    def add_model(self, name: str, config: ModelConfig) -> None:
        """Add or update a model at runtime."""
        self._models[name] = config

    def remove_model(self, name: str) -> bool:
        """Remove a model. Returns True if found and removed."""
        if name in self._models:
            del self._models[name]
            return True
        return False

    def model_exists(self, name: str) -> bool:
        """Check if a model exists."""
        return self.get_model(name) is not None
