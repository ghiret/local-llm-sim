"""Tests for the latency simulator."""

import statistics

import pytest

from local_llm_sim.core.latency import LatencyConfig, LatencySimulator


class TestLatencyConfig:
    """Tests for LatencyConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that default variance parameters are set."""
        config = LatencyConfig(prefill_tps=400, decode_tps=20)

        assert config.prefill_jitter_std == 0.12
        assert config.decode_jitter_std == 0.20
        assert config.stutter_probability == 0.05
        assert config.thermal_decay_start == 200
        assert config.thermal_decay_rate == 0.0003


class TestLatencySimulator:
    """Tests for LatencySimulator class."""

    def test_prefill_delay_calculation(self) -> None:
        """Test that prefill delay is approximately correct."""
        config = LatencyConfig(prefill_tps=400, decode_tps=20)
        simulator = LatencySimulator(config)

        # 4000 tokens at 400 tps = 10 seconds base
        delays = [simulator._calculate_prefill_delay(4000) for _ in range(100)]

        mean_delay = statistics.mean(delays)
        # Should be within 20% of expected 10s (accounting for jitter)
        assert 8 < mean_delay < 12

    def test_prefill_delay_zero_tokens(self) -> None:
        """Test prefill delay with zero tokens."""
        config = LatencyConfig(prefill_tps=400, decode_tps=20)
        simulator = LatencySimulator(config)

        delay = simulator._calculate_prefill_delay(0)
        assert delay == 0

    def test_prefill_delay_zero_tps(self) -> None:
        """Test prefill delay with zero tps (passthrough)."""
        config = LatencyConfig(prefill_tps=0, decode_tps=20)
        simulator = LatencySimulator(config)

        delay = simulator._calculate_prefill_delay(1000)
        assert delay == 0

    def test_prefill_jitter_distribution(self) -> None:
        """Test that prefill delay has variance (not constant)."""
        config = LatencyConfig(prefill_tps=400, decode_tps=20)
        simulator = LatencySimulator(config)

        delays = [simulator._calculate_prefill_delay(1000) for _ in range(1000)]

        # Should have variance (not all the same)
        assert statistics.stdev(delays) > 0.1

    def test_decode_delay_calculation(self) -> None:
        """Test that decode delay is approximately correct."""
        config = LatencyConfig(prefill_tps=400, decode_tps=20)
        simulator = LatencySimulator(config)

        # Base delay should be 1/20 = 0.05 seconds
        delays = [simulator._calculate_token_delay(0)[0] for _ in range(100)]

        mean_delay = statistics.mean(delays)
        # Should be around 0.05s with some variance
        assert 0.03 < mean_delay < 0.08

    def test_decode_stutter_occurs(self) -> None:
        """Test that stutters occur with expected probability."""
        config = LatencyConfig(
            prefill_tps=400,
            decode_tps=20,
            stutter_probability=0.10,  # 10% for easier testing
        )
        simulator = LatencySimulator(config)

        results = [simulator._calculate_token_delay(i) for i in range(1000)]
        stutters = sum(1 for _, is_stutter in results if is_stutter)

        # Expect ~100 stutters (10% of 1000), allow some variance
        assert 50 < stutters < 150

    def test_stutter_multiplier_range(self) -> None:
        """Test that stutter delays are in expected range on average."""
        config = LatencyConfig(
            prefill_tps=400,
            decode_tps=20,
            decode_jitter_std=0.0,  # Disable jitter for this test
            stutter_probability=1.0,  # Always stutter for this test
            stutter_multiplier_min=2.0,
            stutter_multiplier_max=4.0,
        )
        simulator = LatencySimulator(config)

        base_delay = 1 / 20  # 0.05s
        delays = [simulator._calculate_token_delay(0)[0] for _ in range(100)]

        # With no jitter and always stuttering, all delays should be at least 2x base
        for delay in delays:
            assert delay >= base_delay * 2.0

        # Average should be around 3x (midpoint of 2-4x range)
        mean_delay = statistics.mean(delays)
        assert base_delay * 2.5 < mean_delay < base_delay * 3.5

    def test_thermal_decay(self) -> None:
        """Test that thermal decay increases delay over time."""
        config = LatencyConfig(
            prefill_tps=400,
            decode_tps=20,
            stutter_probability=0,  # Disable stutters for this test
            thermal_decay_start=100,
            thermal_decay_rate=0.001,  # Higher rate for testing
        )
        simulator = LatencySimulator(config)

        # Average delay at token 50 vs token 500
        early_delays = [simulator._calculate_token_delay(50)[0] for _ in range(100)]
        late_delays = [simulator._calculate_token_delay(500)[0] for _ in range(100)]

        # Late should be slower on average
        assert statistics.mean(late_delays) > statistics.mean(early_delays)

    def test_thermal_decay_not_applied_before_threshold(self) -> None:
        """Test that thermal decay doesn't apply before threshold."""
        config = LatencyConfig(
            prefill_tps=400,
            decode_tps=20,
            stutter_probability=0,
            thermal_decay_start=200,
        )
        simulator = LatencySimulator(config)

        # Delays at tokens before threshold should be similar
        delays_50 = [simulator._calculate_token_delay(50)[0] for _ in range(100)]
        delays_150 = [simulator._calculate_token_delay(150)[0] for _ in range(100)]

        mean_50 = statistics.mean(delays_50)
        mean_150 = statistics.mean(delays_150)

        # Should be very similar (within variance)
        assert abs(mean_50 - mean_150) < 0.01


@pytest.mark.asyncio
async def test_simulate_prefill() -> None:
    """Test async prefill simulation."""
    config = LatencyConfig(prefill_tps=10000, decode_tps=1000)  # Fast for testing
    simulator = LatencySimulator(config)

    delay = await simulator.simulate_prefill(100)

    # 100 tokens at 10000 tps = 0.01 seconds
    assert 0 < delay < 0.05  # Allow for jitter


@pytest.mark.asyncio
async def test_throttle_stream() -> None:
    """Test stream throttling."""
    config = LatencyConfig(prefill_tps=100000, decode_tps=10000)  # Very fast for testing
    simulator = LatencySimulator(config)

    async def mock_stream():
        for i in range(10):
            yield {"token": i}

    results = []
    final_stats = None
    async for item, stats in simulator.throttle_stream(mock_stream(), input_tokens=100):
        results.append(item)
        final_stats = stats

    assert len(results) == 10
    assert final_stats is not None
    assert final_stats.output_tokens == 10
    assert final_stats.input_tokens == 100


@pytest.mark.asyncio
async def test_throttle_stream_simple() -> None:
    """Test simplified stream throttling."""
    config = LatencyConfig(prefill_tps=100000, decode_tps=10000)
    simulator = LatencySimulator(config)

    async def mock_stream():
        for i in range(5):
            yield {"token": i}

    results = []
    async for item in simulator.throttle_stream_simple(mock_stream(), input_tokens=50):
        results.append(item)

    assert len(results) == 5
    assert results[0] == {"token": 0}
    assert results[4] == {"token": 4}
