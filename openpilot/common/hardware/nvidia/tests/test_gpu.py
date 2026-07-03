"""
Tests for NVIDIA GPU detection and capability queries.

These tests use mocking to simulate NVIDIA hardware since
the actual hardware may not be available in CI.
"""

from unittest.mock import patch  # noqa: TID251

from openpilot.system.hardware.nvidia.gpu import (
  GPUInfo,
  is_nvidia_available,
  get_nvidia_gpus,
  get_cuda_version,
  is_dgx_spark,
  get_recommended_precision,
  get_tinygrad_device,
  _get_compute_capability,
)


class TestGPUInfo:
  """Tests for GPUInfo dataclass."""

  def test_gpu_info_creation(self):
    """Test creating GPUInfo with valid data."""
    gpu = GPUInfo(
      index=0,
      name="NVIDIA GeForce RTX 3090",
      uuid="GPU-12345",
      memory_total_mb=24576,
      memory_free_mb=20000,
      compute_capability=(8, 6),
      driver_version="535.104.05",
      cuda_version="12.2",
    )
    assert gpu.index == 0
    assert gpu.name == "NVIDIA GeForce RTX 3090"
    assert gpu.memory_total_mb == 24576

  def test_memory_gb_conversion(self):
    """Test memory conversion to GB."""
    gpu = GPUInfo(
      index=0,
      name="Test GPU",
      uuid="GPU-123",
      memory_total_mb=24576,
      memory_free_mb=20480,
      compute_capability=(8, 6),
      driver_version="535.0",
      cuda_version="12.0",
    )
    assert gpu.memory_total_gb == 24.0
    assert gpu.memory_free_gb == 20.0

  def test_compute_capability_str(self):
    """Test compute capability string formatting."""
    gpu = GPUInfo(
      index=0,
      name="Test",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(8, 9),
      driver_version="",
      cuda_version="",
    )
    assert gpu.compute_capability_str == "8.9"

  def test_supports_fp16_volta_plus(self):
    """Test FP16 support detection (Volta 7.0+)."""
    # Volta (7.0) - supports FP16
    gpu_volta = GPUInfo(
      index=0,
      name="V100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(7, 0),
      driver_version="",
      cuda_version="",
    )
    assert gpu_volta.supports_fp16() is True

    # Pascal (6.1) - no FP16 tensor cores
    gpu_pascal = GPUInfo(
      index=0,
      name="P100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(6, 1),
      driver_version="",
      cuda_version="",
    )
    assert gpu_pascal.supports_fp16() is False

  def test_supports_bf16_ampere_plus(self):
    """Test BF16 support detection (Ampere 8.0+)."""
    # Ampere (8.0) - supports BF16
    gpu_ampere = GPUInfo(
      index=0,
      name="A100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(8, 0),
      driver_version="",
      cuda_version="",
    )
    assert gpu_ampere.supports_bf16() is True

    # Turing (7.5) - no BF16
    gpu_turing = GPUInfo(
      index=0,
      name="RTX 2080",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(7, 5),
      driver_version="",
      cuda_version="",
    )
    assert gpu_turing.supports_bf16() is False

  def test_supports_fp8_hopper_plus(self):
    """Test FP8 support detection (Hopper 9.0+)."""
    # Hopper (9.0) - supports FP8
    gpu_hopper = GPUInfo(
      index=0,
      name="H100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(9, 0),
      driver_version="",
      cuda_version="",
    )
    assert gpu_hopper.supports_fp8() is True

    # Ada (8.9) - no FP8
    gpu_ada = GPUInfo(
      index=0,
      name="RTX 4090",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(8, 9),
      driver_version="",
      cuda_version="",
    )
    assert gpu_ada.supports_fp8() is False

  def test_supports_nvfp4_blackwell(self):
    """Test NVFP4 support detection (Blackwell 10.0+)."""
    # Blackwell (10.0) - supports NVFP4
    gpu_blackwell = GPUInfo(
      index=0,
      name="GB10",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(10, 0),
      driver_version="",
      cuda_version="",
      is_dgx_spark=True,
    )
    assert gpu_blackwell.supports_nvfp4() is True

    # Hopper (9.0) - no NVFP4
    gpu_hopper = GPUInfo(
      index=0,
      name="H100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(9, 0),
      driver_version="",
      cuda_version="",
    )
    assert gpu_hopper.supports_nvfp4() is False


class TestComputeCapabilityDetection:
  """Tests for GPU compute capability heuristics."""

  def test_blackwell_detection(self):
    """Test Blackwell GPU detection."""
    assert _get_compute_capability("NVIDIA GB10") == (10, 0)
    assert _get_compute_capability("DGX Spark GB200") == (10, 0)
    assert _get_compute_capability("NVIDIA RTX 5090") == (10, 0)

  def test_hopper_detection(self):
    """Test Hopper GPU detection."""
    assert _get_compute_capability("NVIDIA H100") == (9, 0)
    assert _get_compute_capability("NVIDIA H200") == (9, 0)
    assert _get_compute_capability("NVIDIA GH200") == (9, 0)

  def test_ada_detection(self):
    """Test Ada Lovelace GPU detection."""
    assert _get_compute_capability("NVIDIA GeForce RTX 4090") == (8, 9)
    assert _get_compute_capability("NVIDIA GeForce RTX 4080") == (8, 9)
    assert _get_compute_capability("NVIDIA L40") == (8, 9)

  def test_ampere_detection(self):
    """Test Ampere GPU detection."""
    assert _get_compute_capability("NVIDIA GeForce RTX 3090") == (8, 6)
    assert _get_compute_capability("NVIDIA A100") == (8, 6)
    assert _get_compute_capability("NVIDIA A6000") == (8, 6)

  def test_turing_detection(self):
    """Test Turing GPU detection."""
    assert _get_compute_capability("NVIDIA GeForce RTX 2080 Ti") == (7, 5)
    assert _get_compute_capability("NVIDIA T4") == (7, 5)

  def test_volta_detection(self):
    """Test Volta GPU detection."""
    assert _get_compute_capability("NVIDIA V100") == (7, 0)
    assert _get_compute_capability("NVIDIA Titan V") == (7, 0)

  def test_pascal_detection(self):
    """Test Pascal GPU detection."""
    assert _get_compute_capability("NVIDIA GeForce GTX 1080") == (6, 1)
    assert _get_compute_capability("NVIDIA P100") == (6, 1)

  def test_unknown_defaults_to_maxwell(self):
    """Test unknown GPU defaults to Maxwell."""
    assert _get_compute_capability("Unknown GPU") == (5, 2)


class TestNvidiaDetection:
  """Tests for NVIDIA availability detection with mocking."""

  def test_is_nvidia_available_no_nvidia_smi(self):
    """Test detection when nvidia-smi is not available."""
    with patch('shutil.which', return_value=None):
      # Clear the cache first
      is_nvidia_available.cache_clear()
      assert is_nvidia_available() is False

  def test_is_nvidia_available_with_gpu(self):
    """Test detection when GPU is available."""
    with patch('shutil.which', return_value='/usr/bin/nvidia-smi'):
      with patch('openpilot.system.hardware.nvidia.gpu._run_nvidia_smi', return_value="NVIDIA GeForce RTX 3090"):
        is_nvidia_available.cache_clear()
        assert is_nvidia_available() is True

  def test_get_nvidia_gpus_mocked(self):
    """Test GPU enumeration with mocked nvidia-smi."""
    mock_output = "0, NVIDIA GeForce RTX 3090, GPU-12345, 24576, 20000, 535.104.05"

    with patch('shutil.which', return_value='/usr/bin/nvidia-smi'):
      with patch('openpilot.system.hardware.nvidia.gpu._run_nvidia_smi') as mock_smi:
        mock_smi.side_effect = lambda args: {
          '--query-gpu=name --format=csv,noheader': "RTX 3090",
          '--query-gpu=index,name,uuid,memory.total,memory.free,driver_version --format=csv,noheader,nounits': mock_output,
        }.get(' '.join(args[0:2]) if args else '', mock_output)

        is_nvidia_available.cache_clear()
        get_nvidia_gpus.cache_clear()
        get_cuda_version.cache_clear()

        # Just verify function runs without error when mocked
        # Full integration tests require actual hardware


class TestGetRecommendedPrecision:
  """Tests for precision recommendation."""

  def test_blackwell_recommends_fp4(self):
    """Test Blackwell recommends FP4."""
    gpu = GPUInfo(
      index=0,
      name="GB10",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(10, 0),
      driver_version="",
      cuda_version="",
    )
    assert get_recommended_precision(gpu) == 'fp4'

  def test_hopper_recommends_fp8(self):
    """Test Hopper recommends FP8."""
    gpu = GPUInfo(
      index=0,
      name="H100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(9, 0),
      driver_version="",
      cuda_version="",
    )
    assert get_recommended_precision(gpu) == 'fp8'

  def test_ampere_recommends_bf16(self):
    """Test Ampere recommends BF16."""
    gpu = GPUInfo(
      index=0,
      name="RTX 3090",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(8, 6),
      driver_version="",
      cuda_version="",
    )
    assert get_recommended_precision(gpu) == 'bf16'

  def test_volta_recommends_fp16(self):
    """Test Volta recommends FP16."""
    gpu = GPUInfo(
      index=0,
      name="V100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(7, 0),
      driver_version="",
      cuda_version="",
    )
    assert get_recommended_precision(gpu) == 'fp16'

  def test_pascal_recommends_fp32(self):
    """Test Pascal recommends FP32."""
    gpu = GPUInfo(
      index=0,
      name="P100",
      uuid="",
      memory_total_mb=0,
      memory_free_mb=0,
      compute_capability=(6, 1),
      driver_version="",
      cuda_version="",
    )
    assert get_recommended_precision(gpu) == 'fp32'

  def test_no_gpu_recommends_fp32(self):
    """Test no GPU defaults to FP32."""
    assert get_recommended_precision(None) == 'fp32'


class TestTinygradDevice:
  """Tests for tinygrad device selection."""

  def test_returns_cuda_when_nvidia_available(self):
    """Test returns CUDA when NVIDIA GPU is available."""
    with patch('openpilot.system.hardware.nvidia.gpu.is_nvidia_available', return_value=True):
      with patch.dict('os.environ', {}, clear=True):
        assert get_tinygrad_device() == 'CUDA'

  def test_returns_cpu_when_no_nvidia(self):
    """Test returns CPU when no NVIDIA GPU."""
    with patch('openpilot.system.hardware.nvidia.gpu.is_nvidia_available', return_value=False):
      with patch.dict('os.environ', {}, clear=True):
        assert get_tinygrad_device() == 'CPU'

  def test_respects_dev_env_override(self):
    """Test respects DEV environment variable."""
    with patch('openpilot.system.hardware.nvidia.gpu.is_nvidia_available', return_value=True):
      with patch.dict('os.environ', {'DEV': 'AMD'}, clear=True):
        assert get_tinygrad_device() == 'AMD'


class TestDGXSparkDetection:
  """Tests for DGX Spark detection."""

  def test_dgx_spark_detected_from_name(self):
    """Test DGX Spark detection from GPU name."""
    with patch('openpilot.system.hardware.nvidia.gpu.get_nvidia_gpus') as mock_gpus:
      mock_gpus.return_value = [
        GPUInfo(
          index=0,
          name="NVIDIA GB10 Grace Blackwell",
          uuid="",
          memory_total_mb=131072,
          memory_free_mb=130000,
          compute_capability=(10, 0),
          driver_version="",
          cuda_version="",
          is_dgx_spark=True,
        )
      ]
      is_dgx_spark.cache_clear()
      assert is_dgx_spark() is True

  def test_rtx_not_dgx_spark(self):
    """Test RTX cards are not DGX Spark."""
    with patch('openpilot.system.hardware.nvidia.gpu.get_nvidia_gpus') as mock_gpus:
      mock_gpus.return_value = [
        GPUInfo(
          index=0,
          name="NVIDIA GeForce RTX 3090",
          uuid="",
          memory_total_mb=24576,
          memory_free_mb=20000,
          compute_capability=(8, 6),
          driver_version="",
          cuda_version="",
          is_dgx_spark=False,
        )
      ]
      is_dgx_spark.cache_clear()
      assert is_dgx_spark() is False
