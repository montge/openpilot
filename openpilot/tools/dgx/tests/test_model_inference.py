"""Integration tests for NVIDIA GPU model inference.

These tests verify that openpilot models can be loaded and run
on NVIDIA GPUs (including DGX Spark). Tests are skipped if:
- No NVIDIA GPU is available
- Required dependencies (tinygrad, tensorrt) are not installed

Run with pytest:
  pytest openpilot/tools/dgx/tests/test_model_inference.py -v

For DGX hardware validation, also see:
  openpilot/tools/dgx/HARDWARE_TEST_CHECKLIST.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# Skip entire module if no NVIDIA GPU
def pytest_configure(config):
  """Configure pytest with custom markers."""
  config.addinivalue_line("markers", "requires_gpu: mark test as requiring NVIDIA GPU")


def has_nvidia_gpu() -> bool:
  """Check if NVIDIA GPU is available."""
  try:
    from openpilot.common.hardware.nvidia.gpu import is_nvidia_available

    return is_nvidia_available()
  except ImportError:
    return False


# Skip decorator for GPU-required tests
requires_gpu = pytest.mark.skipif(not has_nvidia_gpu(), reason="No NVIDIA GPU available")


class TestGPUDetection:
  """Tests for GPU detection and capabilities."""

  @requires_gpu
  def test_nvidia_available(self):
    """Test NVIDIA GPU is detected."""
    from openpilot.common.hardware.nvidia.gpu import is_nvidia_available

    assert is_nvidia_available()

  @requires_gpu
  def test_gpu_info(self):
    """Test GPU info retrieval."""
    from openpilot.common.hardware.nvidia.gpu import get_nvidia_gpus

    gpus = get_nvidia_gpus()
    assert len(gpus) > 0

    gpu = gpus[0]
    assert gpu.name is not None
    assert gpu.compute_capability[0] >= 7  # At least Volta

  @requires_gpu
  def test_precision_detection(self):
    """Test precision capability detection."""
    from openpilot.common.hardware.nvidia.gpu import get_best_gpu

    gpu = get_best_gpu()
    assert gpu is not None

    # All modern GPUs should support FP16
    assert gpu.supports_fp16

  @requires_gpu
  def test_best_gpu_selection(self):
    """Test best GPU selection."""
    from openpilot.common.hardware.nvidia.gpu import get_best_gpu

    gpu = get_best_gpu()
    assert gpu is not None
    assert gpu.device_id >= 0


class TestModelRunner:
  """Tests for model runner with different backends."""

  @requires_gpu
  def test_runner_initialization(self):
    """Test model runner can be initialized."""
    from openpilot.tools.dgx.model_runner import Backend, ModelRunner, Precision

    runner = ModelRunner(
      model_path="openpilot/selfdrive/modeld/models/driving_supercombo.onnx",
      backend=Backend.CPU,  # Use CPU for init test
      precision=Precision.FP32,
    )
    assert runner.backend == Backend.CPU
    assert runner.precision == Precision.FP32

  def test_precision_to_dtype(self):
    """Test precision to numpy dtype conversion."""
    from openpilot.tools.dgx.model_runner import Precision

    assert Precision.FP32.to_numpy_dtype() == np.float32
    assert Precision.FP16.to_numpy_dtype() == np.float16

  def test_backend_recommendation(self):
    """Test backend recommendation logic."""
    from openpilot.tools.dgx.model_runner import get_recommended_backend

    backend = get_recommended_backend()
    # Should return a valid backend
    assert backend is not None

  @requires_gpu
  def test_precision_recommendation(self):
    """Test precision recommendation based on GPU."""
    from openpilot.tools.dgx.model_runner import Backend, get_recommended_precision

    precision = get_recommended_precision(Backend.TINYGRAD_CUDA)
    # Should recommend at least FP16 for CUDA
    assert precision is not None


class TestTinygradCUDA:
  """Tests for tinygrad CUDA backend."""

  @requires_gpu
  def test_tinygrad_cuda_available(self):
    """Test tinygrad CUDA backend is available."""
    import os

    os.environ["CUDA"] = "1"

    from tinygrad import Device

    # CUDA should be available
    assert "CUDA" in Device._devices or Device.DEFAULT == "CUDA"

  @requires_gpu
  def test_tinygrad_tensor_creation(self):
    """Test creating tensors on CUDA."""
    import os

    os.environ["CUDA"] = "1"

    from tinygrad import Device, Tensor

    Device.DEFAULT = "CUDA"

    # Create tensor
    x = Tensor([1.0, 2.0, 3.0])
    result = x.numpy()

    assert result.shape == (3,)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

  @requires_gpu
  def test_tinygrad_basic_ops(self):
    """Test basic operations on CUDA."""
    import os

    os.environ["CUDA"] = "1"

    from tinygrad import Device, Tensor

    Device.DEFAULT = "CUDA"

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = (a + b).numpy()

    np.testing.assert_array_almost_equal(c, [5.0, 7.0, 9.0])


class TestModelLoading:
  """Tests for loading openpilot models."""

  @pytest.fixture
  def model_dir(self) -> Path:
    """Get model directory."""
    return Path("openpilot/selfdrive/modeld/models")

  def test_model_files_exist(self, model_dir: Path):
    """Test that model files exist."""
    # Check for ONNX files or PKL files
    onnx_files = list(model_dir.glob("*.onnx"))
    pkl_files = list(model_dir.glob("*_tinygrad.pkl"))

    assert len(onnx_files) > 0 or len(pkl_files) > 0, f"No model files found in {model_dir}"

class TestSupercomboContract:
  """Contract tests against the combined driving_supercombo.onnx.

  The driving model's metadata (input_shapes, output_slices, ...) is embedded
  in the ONNX metadata_props — there is no sidecar *_metadata.pkl anymore.
  These run on CPU: metadata extraction only parses the protobuf header, and
  the parse tests use random vectors, never the model itself.
  """

  MODEL_PATH = Path("openpilot/selfdrive/modeld/models/driving_supercombo.onnx")

  EXPECTED_INPUTS = {"img", "big_img", "desire_pulse", "traffic_convention", "features_buffer", "action_t"}
  EXPECTED_SLICES = {
    "plan", "lane_lines", "lane_lines_prob", "road_edges", "lead", "lead_prob",
    "meta", "desire_state", "desire_pred", "pose", "wide_from_device_euler",
    "road_transform", "hidden_state",
  }

  @pytest.fixture
  def metadata(self) -> dict:
    if not self.MODEL_PATH.exists():
      pytest.skip(f"{self.MODEL_PATH} not found")
    from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict

    try:
      return make_metadata_dict(self.MODEL_PATH)
    except Exception as e:
      pytest.skip(f"could not parse embedded metadata (git-lfs pointer not pulled?): {e}")

  @pytest.fixture
  def output_len(self, metadata: dict) -> int:
    return next(iter(metadata["output_shapes"].values()))[-1]

  def test_embedded_metadata_inputs(self, metadata: dict):
    from openpilot.selfdrive.modeld.constants import ModelConstants

    input_shapes = metadata["input_shapes"]
    assert set(input_shapes) == self.EXPECTED_INPUTS
    assert input_shapes["img"] == input_shapes["big_img"]
    assert input_shapes["img"][0] == 1 and input_shapes["img"][1] == 6 * ModelConstants.N_FRAMES
    assert input_shapes["features_buffer"][-1] == ModelConstants.FEATURE_LEN
    assert input_shapes["desire_pulse"][-1] == ModelConstants.DESIRE_LEN
    assert input_shapes["traffic_convention"] == (1, ModelConstants.TRAFFIC_CONVENTION_LEN)

  def test_output_slices_cover_output(self, metadata: dict, output_len: int):
    from openpilot.selfdrive.modeld.constants import ModelConstants

    slices = metadata["output_slices"]
    assert self.EXPECTED_SLICES <= set(slices)
    hidden = slices["hidden_state"]
    assert hidden.stop - hidden.start == ModelConstants.FEATURE_LEN
    for name, sl in slices.items():
      lo, hi, _ = sl.indices(output_len)
      assert 0 <= lo <= hi <= output_len, f"slice {name}={sl} outside output of len {output_len}"

  def test_parse_supercombo_outputs(self, metadata: dict, output_len: int):
    from openpilot.selfdrive.modeld.constants import ModelConstants
    from openpilot.tools.dgx.training.teacher import parse_supercombo_outputs

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((2, output_len), dtype=np.float32)
    parsed = parse_supercombo_outputs(raw, metadata["output_slices"])

    assert parsed["plan"].shape == (2, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert parsed["plan_stds"].shape == parsed["plan"].shape
    assert np.all(parsed["plan_stds"] > 0)
    assert parsed["lane_lines"].shape == (2, ModelConstants.NUM_LANE_LINES, ModelConstants.IDX_N, ModelConstants.LANE_LINES_WIDTH)
    assert parsed["road_edges"].shape == (2, ModelConstants.NUM_ROAD_EDGES, ModelConstants.IDX_N, ModelConstants.ROAD_EDGES_WIDTH)
    assert parsed["desire_state"].shape == (2, ModelConstants.DESIRE_PRED_WIDTH)
    np.testing.assert_allclose(parsed["desire_state"].sum(axis=-1), 1.0, atol=1e-5)  # softmaxed
    # hidden_state passes through unparsed, and parsing must not mutate raw
    np.testing.assert_array_equal(parsed["hidden_state"], raw[:, metadata["output_slices"]["hidden_state"]])

  def test_extract_path_distribution_matches_parser(self, metadata: dict, output_len: int):
    """Torch (student) path extraction agrees with the numpy (teacher) parser."""
    torch = pytest.importorskip("torch")
    from openpilot.selfdrive.modeld.constants import Plan
    from openpilot.tools.dgx.training.teacher import parse_supercombo_outputs
    from openpilot.tools.dgx.training.train import extract_path_distribution

    rng = np.random.default_rng(1)
    raw = rng.standard_normal((2, output_len), dtype=np.float32)

    mean, std = extract_path_distribution(torch.from_numpy(raw.copy()), metadata["output_slices"])
    parsed = parse_supercombo_outputs(raw, metadata["output_slices"])

    assert mean.shape == (2, 1, 33, 3) and std.shape == (2, 1, 33, 3)
    assert bool((std > 0).all())
    np.testing.assert_allclose(mean.numpy()[:, 0], parsed["plan"][:, :, Plan.POSITION], rtol=1e-5)
    np.testing.assert_allclose(std.numpy()[:, 0], parsed["plan_stds"][:, :, Plan.POSITION], rtol=1e-5)


class TestMemoryUtils:
  """Tests for memory utilities."""

  def test_unified_memory_manager_init(self):
    """Test UnifiedMemoryManager initialization."""
    from openpilot.tools.dgx.memory_utils import UnifiedMemoryManager

    manager = UnifiedMemoryManager()
    assert manager is not None

  def test_buffer_allocation(self):
    """Test buffer allocation."""
    from openpilot.tools.dgx.memory_utils import UnifiedMemoryManager

    manager = UnifiedMemoryManager()
    buf = manager.allocate("test", (4, 3, 224, 224), np.float32)

    assert buf.shape == (4, 3, 224, 224)
    assert buf.dtype == np.float32

    manager.release("test")

  def test_memory_scope(self):
    """Test memory scope context manager."""
    from openpilot.tools.dgx.memory_utils import memory_scope

    with memory_scope("test") as mem:
      buf1 = mem.allocate("input", (1, 256))
      buf2 = mem.allocate("output", (1, 128))

      assert buf1 is not None
      assert buf2 is not None

    # Buffers should be freed after context

  def test_optimal_batch_size(self):
    """Test optimal batch size calculation."""
    from openpilot.tools.dgx.memory_utils import get_optimal_batch_size

    batch_size = get_optimal_batch_size(
      model_memory_mb=500,
      input_size_mb=10,
      available_memory_mb=8192,
    )

    assert batch_size > 0
    assert batch_size < 1000  # Reasonable upper bound


class TestGPUMonitor:
  """Tests for GPU monitoring utilities."""

  @requires_gpu
  def test_gpu_status(self):
    """Test getting GPU status."""
    from openpilot.tools.dgx.gpu_monitor import get_gpu_status

    status = get_gpu_status()
    assert status is not None
    assert status.name is not None
    assert status.memory is not None

  @requires_gpu
  def test_gpu_memory(self):
    """Test getting GPU memory info."""
    from openpilot.tools.dgx.gpu_monitor import get_gpu_memory

    memory = get_gpu_memory()
    assert memory is not None
    # Memory values should be non-negative
    assert memory.total_mb >= 0
    assert memory.used_mb >= 0

  @requires_gpu
  def test_memory_tracker(self):
    """Test memory tracker context manager."""
    from openpilot.tools.dgx.gpu_monitor import MemoryTracker

    with MemoryTracker("test"):
      # Just verify it doesn't crash
      pass


class TestAlgorithmHarnessGPU:
  """Tests for GPU-accelerated algorithm harness."""

  def test_gpu_scenario_runner_init(self):
    """Test GPUScenarioRunner initialization."""
    from openpilot.tools.dgx.algorithm_harness_gpu import GPUScenarioRunner

    runner = GPUScenarioRunner(max_workers=4)
    assert runner.max_workers == 4

  def test_gpu_metrics_accelerator(self):
    """Test GPUMetricsAccelerator."""
    from openpilot.tools.dgx.algorithm_harness_gpu import GPUMetricsAccelerator

    accelerator = GPUMetricsAccelerator()

    # Test percentile computation
    data = np.random.randn(1000)
    p50, p95, p99 = accelerator.percentiles(data, [50, 95, 99])

    assert p50 < p95 < p99  # Percentiles should be ordered

  def test_batch_result_dataclass(self):
    """Test BatchResult dataclass."""
    from openpilot.tools.dgx.algorithm_harness_gpu import BatchResult

    result = BatchResult(
      results=[],
      total_time_s=1.0,
      scenarios_per_second=100.0,
      num_scenarios=100,
      num_workers=4,
    )

    assert result.total_time_s == 1.0
    assert result.num_workers == 4
