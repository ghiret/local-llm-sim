"""Tests for API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """Test the root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "local-llm-sim"


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """Test the health endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_list_tags(client: AsyncClient) -> None:
    """Test listing available models."""
    response = await client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 2

    names = [m["name"] for m in data["models"]]
    assert "test-model:latest" in names
    assert "embed-model:latest" in names


@pytest.mark.asyncio
async def test_show_model(client: AsyncClient) -> None:
    """Test getting model details."""
    response = await client.post("/api/show", json={"name": "test-model:latest"})
    assert response.status_code == 200
    data = response.json()

    assert "modelfile" in data
    assert "model_info" in data
    assert data["model_info"]["latency"]["prefill_tps"] == 1000
    assert data["model_info"]["latency"]["decode_tps"] == 100
    assert data["model_info"]["capabilities"]["tools"] is True


@pytest.mark.asyncio
async def test_show_model_partial_name(client: AsyncClient) -> None:
    """Test getting model details with partial name."""
    response = await client.post("/api/show", json={"name": "test-model"})
    assert response.status_code == 200
    data = response.json()
    assert "model_info" in data


@pytest.mark.asyncio
async def test_show_model_not_found(client: AsyncClient) -> None:
    """Test getting a non-existent model."""
    response = await client.post("/api/show", json={"name": "nonexistent"})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_pull_model_exists(client: AsyncClient) -> None:
    """Test pulling an existing model."""
    response = await client.post(
        "/api/pull",
        json={"name": "test-model:latest", "stream": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


@pytest.mark.asyncio
async def test_pull_model_not_found(client: AsyncClient) -> None:
    """Test pulling a non-existent model."""
    response = await client.post(
        "/api/pull",
        json={"name": "nonexistent", "stream": False},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_model(client: AsyncClient) -> None:
    """Test deleting a model."""
    # First verify it exists
    response = await client.get("/api/tags")
    initial_count = len(response.json()["models"])

    # Delete it
    response = await client.request(
        "DELETE",
        "/api/delete",
        json={"name": "test-model:latest"},
    )
    assert response.status_code == 200

    # Verify it's gone
    response = await client.get("/api/tags")
    assert len(response.json()["models"]) == initial_count - 1


@pytest.mark.asyncio
async def test_delete_model_not_found(client: AsyncClient) -> None:
    """Test deleting a non-existent model."""
    response = await client.request(
        "DELETE",
        "/api/delete",
        json={"name": "nonexistent"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_running_models(client: AsyncClient) -> None:
    """Test listing running models."""
    response = await client.get("/api/ps")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    # All models should be listed as "running"
    assert len(data["models"]) == 2


@pytest.mark.asyncio
async def test_stats_endpoint(client: AsyncClient) -> None:
    """Test the stats endpoint."""
    response = await client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()

    assert "session" in data
    assert "latency_comparison" in data
    assert "by_model" in data

    assert "started" in data["session"]
    assert "requests" in data["session"]
    assert "simulated_total_sec" in data["latency_comparison"]


@pytest.mark.asyncio
async def test_stats_requests_endpoint(client: AsyncClient) -> None:
    """Test the request log endpoint."""
    response = await client.get("/api/stats/requests")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_stats_reset_endpoint(client: AsyncClient) -> None:
    """Test resetting stats."""
    response = await client.post("/api/stats/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reset"


@pytest.mark.asyncio
async def test_prometheus_metrics(client: AsyncClient) -> None:
    """Test Prometheus metrics endpoint."""
    response = await client.get("/metrics")
    assert response.status_code == 200
    text = response.text

    assert "llm_sim_requests_total" in text
    assert "llm_sim_input_tokens_total" in text
    assert "llm_sim_output_tokens_total" in text
    assert "llm_sim_simulated_seconds_total" in text
    assert "llm_sim_slowdown_factor" in text
