"""Latency simulation for realistic local LLM inference feel."""

import asyncio
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class LatencyConfig:
    """Configuration for latency simulation."""

    prefill_tps: float
    decode_tps: float

    # Variance parameters
    prefill_jitter_std: float = 0.12  # 12% standard deviation
    decode_jitter_std: float = 0.20  # 20% standard deviation
    stutter_probability: float = 0.05  # 5% chance per token
    stutter_multiplier_min: float = 2.0
    stutter_multiplier_max: float = 4.0
    thermal_decay_start: int = 200  # Start slowing after this many tokens
    thermal_decay_rate: float = 0.0003  # Slowdown per token after threshold


@dataclass
class LatencyStats:
    """Statistics from a single request's latency simulation."""

    input_tokens: int = 0
    output_tokens: int = 0
    prefill_delay_ms: float = 0
    decode_delay_ms: float = 0
    total_delay_ms: float = 0
    effective_prefill_tps: float = 0
    effective_decode_tps: float = 0
    stutters: int = 0
    simulated_sleep_ms: float = 0  # Total artificial delay added


class LatencySimulator:
    """
    Simulates realistic local LLM inference latency.

    Models two phases:
    1. Prefill: Processing input tokens (time to first output token)
    2. Decode: Generating output tokens (streaming speed)

    Includes realistic variance:
    - Gaussian jitter on base rates
    - Random stutters (occasional longer delays)
    - Thermal throttling (gradual slowdown on long generations)
    """

    def __init__(self, config: LatencyConfig) -> None:
        self.config = config

    def _gaussian_factor(self, std: float) -> float:
        """Return a multiplier based on gaussian distribution."""
        return 1 + random.gauss(0, std)

    def _calculate_prefill_delay(self, input_tokens: int) -> float:
        """Calculate prefill delay in seconds with jitter."""
        if self.config.prefill_tps <= 0:
            return 0

        base_delay = input_tokens / self.config.prefill_tps
        jitter = self._gaussian_factor(self.config.prefill_jitter_std)
        return max(0, base_delay * jitter)

    def _calculate_token_delay(self, token_index: int) -> tuple[float, bool]:
        """
        Calculate delay for a single token with all variance factors.

        Returns (delay_seconds, is_stutter).
        """
        if self.config.decode_tps <= 0:
            return 0, False

        base_delay = 1 / self.config.decode_tps

        # Gaussian jitter
        jitter = self._gaussian_factor(self.config.decode_jitter_std)

        # Stutter (occasional longer delay)
        is_stutter = random.random() < self.config.stutter_probability
        if is_stutter:
            stutter = random.uniform(
                self.config.stutter_multiplier_min,
                self.config.stutter_multiplier_max,
            )
        else:
            stutter = 1.0

        # Thermal decay (gradual slowdown after threshold)
        if token_index > self.config.thermal_decay_start:
            thermal = (
                1 + (token_index - self.config.thermal_decay_start) * self.config.thermal_decay_rate
            )
        else:
            thermal = 1.0

        delay = max(0, base_delay * jitter * stutter * thermal)
        return delay, is_stutter

    async def simulate_prefill(self, input_tokens: int) -> tuple[float, float]:
        """
        Simulate prefill phase (prompt processing).

        Returns (total_delay_seconds, sleep_seconds).
        """
        delay = self._calculate_prefill_delay(input_tokens)
        if delay > 0:
            await asyncio.sleep(delay)
        return delay, delay  # For prefill, all delay is simulated sleep

    async def throttle_stream(
        self,
        stream: AsyncIterator[T],
        input_tokens: int,
    ) -> AsyncIterator[tuple[T, LatencyStats]]:
        """
        Wrap a stream with latency simulation.

        Yields (item, stats) tuples. Stats are updated incrementally
        and the final yield contains complete statistics.
        """
        stats = LatencyStats(input_tokens=input_tokens)

        # Prefill phase
        prefill_delay, prefill_sleep = await self.simulate_prefill(input_tokens)
        stats.prefill_delay_ms = prefill_delay * 1000
        stats.simulated_sleep_ms = prefill_sleep * 1000

        # Decode phase
        decode_start = time.monotonic()
        token_index = 0

        async for item in stream:
            token_delay, is_stutter = self._calculate_token_delay(token_index)

            if is_stutter:
                stats.stutters += 1

            if token_delay > 0:
                await asyncio.sleep(token_delay)
                stats.simulated_sleep_ms += token_delay * 1000

            token_index += 1
            stats.output_tokens = token_index
            stats.decode_delay_ms = (time.monotonic() - decode_start) * 1000
            stats.total_delay_ms = stats.prefill_delay_ms + stats.decode_delay_ms

            if stats.prefill_delay_ms > 0:
                stats.effective_prefill_tps = input_tokens / (stats.prefill_delay_ms / 1000)
            if stats.decode_delay_ms > 0:
                stats.effective_decode_tps = token_index / (stats.decode_delay_ms / 1000)

            yield item, stats

    async def throttle_stream_simple(
        self,
        stream: AsyncIterator[T],
        input_tokens: int,
    ) -> AsyncIterator[T]:
        """Simplified version that just yields items without stats."""
        async for item, _ in self.throttle_stream(stream, input_tokens):
            yield item
