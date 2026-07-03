"""Tests for FAIR profiling utilities."""

import numpy as np
import pytest

from openpilot.tools.fair.profiling.benchmark import (
  LatencyResult,
  MemoryResult,
  ThroughputResult,
  ModelProfile,
  profile_latency,
  profile_memory,
  benchmark_throughput,
  profile_model,
  compare_models,
  format_comparison_table,
)
from openpilot.tools.fair.profiling.flops import (
  ModelStats,
  count_parameters,
  estimate_flops,
  get_model_stats,
  format_layer_flops,
)

# Check PyTorch availability
try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


class TestLatencyResult:
  """Tests for LatencyResult."""

  def test_creation(self):
    """Test result creation."""
    result = LatencyResult(
      mean_ms=10.5,
      std_ms=1.2,
      min_ms=8.0,
      max_ms=15.0,
      percentiles={"p50": 10.0, "p95": 13.0, "p99": 14.5},
      num_runs=100,
    )
    assert result.mean_ms == 10.5
    assert result.std_ms == 1.2
    assert result.percentiles["p95"] == 13.0

  def test_str(self):
    """Test string formatting."""
    result = LatencyResult(
      mean_ms=10.0,
      std_ms=1.0,
      min_ms=8.0,
      max_ms=15.0,
      percentiles={"p50": 10.0, "p95": 12.0},
    )
    s = str(result)
    assert "10.00" in s
    assert "ms" in s


class TestMemoryResult:
  """Tests for MemoryResult."""

  def test_creation(self):
    """Test result creation."""
    result = MemoryResult(
      peak_mb=512.0,
      allocated_mb=256.0,
      reserved_mb=1024.0,
      model_mb=50.0,
    )
    assert result.peak_mb == 512.0
    assert result.model_mb == 50.0


class TestThroughputResult:
  """Tests for ThroughputResult."""

  def test_creation(self):
    """Test result creation."""
    result = ThroughputResult(
      samples_per_second=1000.0,
      batches_per_second=100.0,
      batch_size=10,
      total_samples=5000,
      total_time_s=5.0,
    )
    assert result.samples_per_second == 1000.0
    assert result.batch_size == 10


class TestModelProfile:
  """Tests for ModelProfile."""

  def test_creation(self):
    """Test profile creation."""
    latency = LatencyResult(10.0, 1.0, 8.0, 12.0)
    memory = MemoryResult(100.0, 50.0)

    profile = ModelProfile(
      name="TestModel",
      latency=latency,
      memory=memory,
      device="cpu",
    )
    assert profile.name == "TestModel"
    assert profile.device == "cpu"

  def test_str(self):
    """Test string formatting."""
    profile = ModelProfile(
      name="TestModel",
      latency=LatencyResult(10.0, 1.0, 8.0, 12.0, {"p50": 10.0, "p95": 11.5}),
      memory=MemoryResult(100.0, 50.0),
      input_shape=(1, 3, 224, 224),
      device="cuda",
    )
    s = str(profile)
    assert "TestModel" in s
    assert "cuda" in s


class TestModelStats:
  """Tests for ModelStats."""

  def test_creation(self):
    """Test stats creation."""
    stats = ModelStats(
      total_params=10_000_000,
      trainable_params=8_000_000,
      total_flops=5_000_000_000,
      param_bytes=40_000_000,
    )
    assert stats.total_params_m == 10.0
    assert stats.trainable_params == 8_000_000
    assert stats.total_flops_g == 5.0

  def test_properties(self):
    """Test computed properties."""
    stats = ModelStats(
      total_params=1_000_000,
      trainable_params=500_000,
      total_flops=1_000_000_000,
      param_bytes=4_000_000,
    )
    assert stats.total_params_m == 1.0
    assert stats.total_flops_g == 1.0
    assert abs(stats.param_mb - 3.81) < 0.1  # ~4MB


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestProfileLatency:
  """Tests for profile_latency."""

  def test_simple_model(self):
    """Test latency profiling on simple model."""
    model = nn.Linear(64, 32)
    model.eval()

    input_data = torch.randn(8, 64)
    result = profile_latency(model, input_data, num_warmup=5, num_runs=20)

    assert result.mean_ms > 0
    assert result.std_ms >= 0
    assert result.num_runs == 20

  def test_with_numpy_input(self):
    """Test with numpy input."""
    model = nn.Linear(32, 16)
    model.eval()

    input_data = np.random.randn(4, 32).astype(np.float32)
    result = profile_latency(model, input_data, num_warmup=2, num_runs=10)

    assert result.mean_ms > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestProfileMemory:
  """Tests for profile_memory."""

  def test_cpu_memory(self):
    """Test CPU memory profiling."""
    model = nn.Linear(256, 128)

    input_data = torch.randn(16, 256)
    result = profile_memory(model, input_data, device="cpu")

    assert result.peak_mb > 0
    assert result.model_mb > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBenchmarkThroughput:
  """Tests for benchmark_throughput."""

  def test_simple_model(self):
    """Test throughput benchmarking."""
    model = nn.Sequential(
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
    )
    model.eval()

    results = benchmark_throughput(
      model,
      input_shape=(64,),
      batch_sizes=[1, 4],
      duration_s=0.5,
      device="cpu",
    )

    assert 1 in results
    assert 4 in results
    assert results[1].samples_per_second > 0
    assert results[4].samples_per_second > results[1].samples_per_second


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestProfileModel:
  """Tests for profile_model."""

  def test_complete_profile(self):
    """Test complete model profiling."""
    model = nn.Sequential(
      nn.Conv2d(3, 16, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, padding=1),
      nn.ReLU(),
    )
    model.eval()

    profile = profile_model(
      model,
      input_shape=(3, 32, 32),
      batch_size=2,
      device="cpu",
      num_latency_runs=10,
      name="SimpleConvNet",
    )

    assert profile.name == "SimpleConvNet"
    assert profile.latency is not None
    assert profile.memory is not None
    assert profile.input_shape == (2, 3, 32, 32)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCompareModels:
  """Tests for compare_models."""

  def test_comparison(self):
    """Test model comparison."""
    models = {
      "small": nn.Linear(64, 32),
      "large": nn.Linear(64, 128),
    }
    for m in models.values():
      m.eval()

    profiles = compare_models(
      models,
      input_shape=(64,),
      batch_size=1,
      device="cpu",
    )

    assert "small" in profiles
    assert "large" in profiles

  def test_format_table(self):
    """Test table formatting."""
    profiles = {
      "model1": ModelProfile(
        name="model1",
        latency=LatencyResult(10.0, 1.0, 8.0, 12.0, {"p50": 10.0, "p95": 11.0}),
        memory=MemoryResult(100.0, 50.0),
        throughput=ThroughputResult(500.0, 100.0, 5, 2500, 5.0),
      ),
    }
    table = format_comparison_table(profiles)
    assert "model1" in table
    assert "Latency" in table


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCountParameters:
  """Tests for count_parameters."""

  def test_linear(self):
    """Test parameter counting for linear layer."""
    model = nn.Linear(100, 50, bias=True)
    total, trainable = count_parameters(model)

    # 100*50 + 50 = 5050
    assert total == 5050
    assert trainable == 5050

  def test_frozen_params(self):
    """Test counting with frozen parameters."""
    model = nn.Linear(64, 32)
    model.weight.requires_grad = False

    total, trainable = count_parameters(model)

    assert total == 64 * 32 + 32  # weight + bias
    assert trainable == 32  # only bias


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEstimateFlops:
  """Tests for estimate_flops."""

  def test_linear_flops(self):
    """Test FLOPs for linear layer."""
    model = nn.Linear(128, 64)
    model.eval()

    flops = estimate_flops(model, input_shape=(1, 128))

    # 2 * 128 * 64 + 64 (bias) = 16448
    assert flops > 0
    assert flops == 2 * 128 * 64 + 64

  def test_conv_flops(self):
    """Test FLOPs for conv layer."""
    model = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    model.eval()

    flops = estimate_flops(model, input_shape=(1, 3, 32, 32))

    # FLOPs should be significant
    assert flops > 100000

  def test_detailed(self):
    """Test detailed FLOPs breakdown."""
    model = nn.Sequential(
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 16),
    )
    model.eval()

    layer_flops = estimate_flops(model, input_shape=(1, 64), detailed=True)

    assert isinstance(layer_flops, dict)
    assert len(layer_flops) >= 2  # At least 2 linear layers


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGetModelStats:
  """Tests for get_model_stats."""

  def test_complete_stats(self):
    """Test complete model statistics."""
    model = nn.Sequential(
      nn.Conv2d(3, 32, 3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(32 * 30 * 30, 64),
    )
    model.eval()

    stats = get_model_stats(model, input_shape=(1, 3, 32, 32))

    assert stats.total_params > 0
    assert stats.total_flops > 0
    assert stats.param_bytes > 0


class TestFormatLayerFlops:
  """Tests for format_layer_flops."""

  def test_formatting(self):
    """Test layer FLOPs formatting."""
    layer_flops = {
      "conv1": 10_000_000,
      "conv2": 20_000_000,
      "fc": 5_000_000,
    }

    table = format_layer_flops(layer_flops, top_k=10)

    assert "conv2" in table
    assert "Total" in table
    assert "FLOPs" in table

  def test_top_k_limit(self):
    """Test top_k limiting."""
    layer_flops = {f"layer{i}": i * 1000 for i in range(50)}

    table = format_layer_flops(layer_flops, top_k=5)

    assert "layer49" in table  # Highest
    assert "and 45 more layers" in table
