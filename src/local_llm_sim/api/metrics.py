"""Metrics and statistics API endpoints."""

import threading
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

router = APIRouter(tags=["metrics"])


class RequestRecord(BaseModel):
    """Record of a single request's performance."""

    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    simulated_prefill_ms: float
    simulated_decode_ms: float
    simulated_total_ms: float
    backend_ms: float
    effective_prefill_tps: float
    effective_decode_tps: float


class SessionStats(BaseModel):
    """Session-level statistics."""

    started: str
    requests: int
    input_tokens: int
    output_tokens: int
    openrouter_cost_usd: float


class LatencyComparison(BaseModel):
    """Comparison of simulated vs backend latency."""

    simulated_total_sec: float
    backend_total_sec: float
    slowdown_factor: float


class StatsResponse(BaseModel):
    """Full statistics response."""

    session: SessionStats
    latency_comparison: LatencyComparison
    by_model: dict[str, dict[str, Any]]


class MetricsStore:
    """Thread-safe storage for request metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = datetime.now(UTC)
        self._requests: list[RequestRecord] = []
        self._total_cost = 0.0

    def record_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        simulated_prefill_ms: float,
        simulated_decode_ms: float,
        backend_ms: float,
        cost_usd: float = 0.0,
    ) -> None:
        """Record a completed request."""
        with self._lock:
            prefill_tps = (
                input_tokens / (simulated_prefill_ms / 1000) if simulated_prefill_ms > 0 else 0
            )
            decode_tps = (
                output_tokens / (simulated_decode_ms / 1000) if simulated_decode_ms > 0 else 0
            )

            self._requests.append(
                RequestRecord(
                    timestamp=datetime.now(UTC).isoformat(),
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    simulated_prefill_ms=simulated_prefill_ms,
                    simulated_decode_ms=simulated_decode_ms,
                    simulated_total_ms=simulated_prefill_ms + simulated_decode_ms,
                    backend_ms=backend_ms,
                    effective_prefill_tps=prefill_tps,
                    effective_decode_tps=decode_tps,
                )
            )
            self._total_cost += cost_usd

    def get_stats(self) -> StatsResponse:
        """Get aggregated statistics."""
        with self._lock:
            total_input = sum(r.input_tokens for r in self._requests)
            total_output = sum(r.output_tokens for r in self._requests)
            total_simulated_ms = sum(r.simulated_total_ms for r in self._requests)
            total_backend_ms = sum(r.backend_ms for r in self._requests)

            # Per-model breakdown
            by_model: dict[str, dict[str, Any]] = {}
            for r in self._requests:
                if r.model not in by_model:
                    by_model[r.model] = {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "simulated_ms": 0,
                        "backend_ms": 0,
                    }
                by_model[r.model]["requests"] += 1
                by_model[r.model]["input_tokens"] += r.input_tokens
                by_model[r.model]["output_tokens"] += r.output_tokens
                by_model[r.model]["simulated_ms"] += r.simulated_total_ms
                by_model[r.model]["backend_ms"] += r.backend_ms

            return StatsResponse(
                session=SessionStats(
                    started=self._started.isoformat(),
                    requests=len(self._requests),
                    input_tokens=total_input,
                    output_tokens=total_output,
                    openrouter_cost_usd=self._total_cost,
                ),
                latency_comparison=LatencyComparison(
                    simulated_total_sec=total_simulated_ms / 1000,
                    backend_total_sec=total_backend_ms / 1000,
                    slowdown_factor=(
                        total_simulated_ms / total_backend_ms if total_backend_ms > 0 else 0
                    ),
                ),
                by_model=by_model,
            )

    def get_recent_requests(self, limit: int = 100) -> list[RequestRecord]:
        """Get recent request records."""
        with self._lock:
            return list(reversed(self._requests[-limit:]))

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._started = datetime.now(UTC)
            self._requests = []
            self._total_cost = 0.0


# Global metrics instance
metrics = MetricsStore()


@router.get("/api/stats")
async def get_stats() -> StatsResponse:
    """Get session statistics."""
    return metrics.get_stats()


@router.get("/api/stats/requests")
async def get_requests(limit: int = 100) -> list[RequestRecord]:
    """Get recent request records."""
    return metrics.get_recent_requests(limit)


@router.post("/api/stats/reset")
async def reset_stats() -> dict[str, str]:
    """Reset all statistics."""
    metrics.reset()
    return {"status": "reset"}


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Prometheus-format metrics."""
    stats = metrics.get_stats()

    lines = [
        "# HELP llm_sim_requests_total Total number of requests",
        "# TYPE llm_sim_requests_total counter",
        f"llm_sim_requests_total {stats.session.requests}",
        "",
        "# HELP llm_sim_input_tokens_total Total input tokens processed",
        "# TYPE llm_sim_input_tokens_total counter",
        f"llm_sim_input_tokens_total {stats.session.input_tokens}",
        "",
        "# HELP llm_sim_output_tokens_total Total output tokens generated",
        "# TYPE llm_sim_output_tokens_total counter",
        f"llm_sim_output_tokens_total {stats.session.output_tokens}",
        "",
        "# HELP llm_sim_simulated_seconds_total Total simulated latency",
        "# TYPE llm_sim_simulated_seconds_total counter",
        f"llm_sim_simulated_seconds_total {stats.latency_comparison.simulated_total_sec}",
        "",
        "# HELP llm_sim_backend_seconds_total Total backend latency",
        "# TYPE llm_sim_backend_seconds_total counter",
        f"llm_sim_backend_seconds_total {stats.latency_comparison.backend_total_sec}",
        "",
        "# HELP llm_sim_slowdown_factor Current slowdown factor",
        "# TYPE llm_sim_slowdown_factor gauge",
        f"llm_sim_slowdown_factor {stats.latency_comparison.slowdown_factor}",
        "",
        "# HELP llm_sim_cost_usd_total Total OpenRouter cost",
        "# TYPE llm_sim_cost_usd_total counter",
        f"llm_sim_cost_usd_total {stats.session.openrouter_cost_usd}",
    ]

    return "\n".join(lines)
