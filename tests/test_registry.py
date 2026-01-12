"""Tests for the model registry."""

import tempfile
from pathlib import Path

from local_llm_sim.core.registry import ModelRegistry
from local_llm_sim.models.config import CapabilitiesConfig, LatencyConfig, ModelConfig


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_load_from_yaml(self, temp_config: Path) -> None:
        """Test loading models from YAML config."""
        registry = ModelRegistry(temp_config)

        # Should have loaded the test models
        assert registry.model_exists("test-model:latest")
        assert registry.model_exists("embed-model:latest")

    def test_get_model_exact_match(self, temp_config: Path) -> None:
        """Test getting a model by exact name."""
        registry = ModelRegistry(temp_config)

        model = registry.get_model("test-model:latest")
        assert model is not None
        assert model.backend == "openai/gpt-3.5-turbo"
        assert model.latency.prefill_tps == 1000
        assert model.latency.decode_tps == 100

    def test_get_model_partial_match(self, temp_config: Path) -> None:
        """Test getting a model by partial name (without tag)."""
        registry = ModelRegistry(temp_config)

        # Should match "test-model:latest"
        model = registry.get_model("test-model")
        assert model is not None
        assert model.backend == "openai/gpt-3.5-turbo"

    def test_get_model_not_found(self, temp_config: Path) -> None:
        """Test getting a non-existent model."""
        registry = ModelRegistry(temp_config)

        model = registry.get_model("nonexistent-model")
        assert model is None

    def test_list_models(self, temp_config: Path) -> None:
        """Test listing all models."""
        registry = ModelRegistry(temp_config)

        models = registry.list_models()
        assert len(models) == 2

        names = [m.name for m in models]
        assert "test-model:latest" in names
        assert "embed-model:latest" in names

    def test_model_info_format(self, temp_config: Path) -> None:
        """Test that model info is in correct Ollama format."""
        registry = ModelRegistry(temp_config)

        models = registry.list_models()
        test_model = next(m for m in models if m.name == "test-model:latest")

        assert test_model.name == "test-model:latest"
        assert test_model.model == "test-model:latest"
        assert test_model.size == 1000000
        assert len(test_model.digest) == 64  # SHA256 hex
        assert test_model.details.parameter_size == "7B"
        assert test_model.details.quantization_level == "Q4_K_M"

    def test_add_model(self, temp_config: Path) -> None:
        """Test adding a model at runtime."""
        registry = ModelRegistry(temp_config)

        new_config = ModelConfig(
            backend="anthropic/claude-3-haiku",
            display_name="Claude Haiku",
            latency=LatencyConfig(prefill_tps=500, decode_tps=50),
            capabilities=CapabilitiesConfig(tools=True),
        )

        registry.add_model("claude-haiku:latest", new_config)

        assert registry.model_exists("claude-haiku:latest")
        model = registry.get_model("claude-haiku:latest")
        assert model is not None
        assert model.backend == "anthropic/claude-3-haiku"

    def test_remove_model(self, temp_config: Path) -> None:
        """Test removing a model."""
        registry = ModelRegistry(temp_config)

        assert registry.model_exists("test-model:latest")
        result = registry.remove_model("test-model:latest")
        assert result is True
        assert not registry.model_exists("test-model:latest")

    def test_remove_model_not_found(self, temp_config: Path) -> None:
        """Test removing a non-existent model."""
        registry = ModelRegistry(temp_config)

        result = registry.remove_model("nonexistent")
        assert result is False

    def test_capabilities_loaded(self, temp_config: Path) -> None:
        """Test that capabilities are loaded correctly."""
        registry = ModelRegistry(temp_config)

        test_model = registry.get_model("test-model:latest")
        assert test_model is not None
        assert test_model.capabilities.tools is True
        assert test_model.capabilities.vision is False
        assert test_model.capabilities.embeddings is False

        embed_model = registry.get_model("embed-model:latest")
        assert embed_model is not None
        assert embed_model.capabilities.embeddings is True

    def test_empty_config_directory(self) -> None:
        """Test handling of empty config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(Path(tmpdir))
            models = registry.list_models()
            assert models == []

    def test_missing_config_directory(self) -> None:
        """Test handling of missing config directory."""
        registry = ModelRegistry(Path("/nonexistent/path"))
        models = registry.list_models()
        assert models == []

    def test_get_model_name(self, temp_config: Path) -> None:
        """Test getting the full model name for partial matches."""
        registry = ModelRegistry(temp_config)

        # Exact match
        name = registry.get_model_name("test-model:latest")
        assert name == "test-model:latest"

        # Partial match
        name = registry.get_model_name("test-model")
        assert name == "test-model:latest"

        # Not found
        name = registry.get_model_name("nonexistent")
        assert name is None
