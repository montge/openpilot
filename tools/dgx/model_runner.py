"""Model runner with precision selection and profiling for NVIDIA GPUs.

Provides a unified interface for running openpilot models with:
- Backend selection (TensorRT, tinygrad CUDA, CPU)
- Precision modes (FP32, FP16, BF16, FP8)
- Warm-up runs for accurate benchmarking
- Performance profiling and memory tracking

This is the runtime component that loads pre-compiled models.
For model compilation with different precisions, see compile_models.py.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Precision(Enum):
  """Model precision modes."""

  FP32 = "fp32"
  FP16 = "fp16"
  BF16 = "bf16"
  FP8 = "fp8"
  FP4 = "fp4"

  def to_numpy_dtype(self) -> np.dtype:
    """Get numpy dtype for this precision."""
    if self in (Precision.FP32, Precision.FP8, Precision.FP4):
      # FP8/FP4 compute internally but use FP32 for I/O
      return np.dtype(np.float32)
    elif self == Precision.FP16:
      return np.dtype(np.float16)
    elif self == Precision.BF16:
      # NumPy doesn't have bfloat16, use float32 for I/O
      return np.dtype(np.float32)
    return np.dtype(np.float32)


class Backend(Enum):
  """Model execution backends."""

  TENSORRT = "tensorrt"
  TINYGRAD_CUDA = "tinygrad_cuda"
  TINYGRAD_CPU = "tinygrad_cpu"
  CPU = "cpu"


@dataclass
class InferenceStats:
  """Statistics from model inference."""

  inference_time_ms: float
  throughput_fps: float
  memory_used_mb: float = 0.0
  warmup_time_ms: float = 0.0
  backend: str = ""
  precision: str = ""


@dataclass
class ProfilingResult:
  """Detailed profiling results."""

  model_name: str
  backend: Backend
  precision: Precision
  runs: int
  warmup_runs: int

  # Timing stats (in milliseconds)
  mean_time_ms: float
  std_time_ms: float
  min_time_ms: float
  max_time_ms: float
  p50_time_ms: float
  p95_time_ms: float
  p99_time_ms: float

  # Throughput
  mean_fps: float

  # Memory (if available)
  peak_memory_mb: float = 0.0

  # Raw timings for analysis
  timings_ms: list[float] = field(default_factory=list)


class ModelRunner:
  """Unified model runner with precision and backend selection.

  Supports multiple backends:
  - TensorRT: Fastest inference, requires tensorrt package
  - tinygrad CUDA: Uses tinygrad with CUDA backend
  - CPU: Fallback for testing without GPU

  Usage:
    runner = ModelRunner(
        model_path="selfdrive/modeld/models/driving_policy.onnx",
        backend=Backend.TENSORRT,
        precision=Precision.FP16,
    )
    runner.warmup(runs=10)
    output = runner.run(inputs)
    stats = runner.profile(inputs, runs=100)
  """

  def __init__(
    self,
    model_path: str | Path,
    backend: Backend = Backend.TENSORRT,
    precision: Precision = Precision.FP16,
    device_id: int = 0,
  ):
    """Initialize model runner.

    Args:
      model_path: Path to model file (.onnx or .pkl)
      backend: Execution backend to use
      precision: Precision mode for inference
      device_id: GPU device ID (for multi-GPU systems)
    """
    self.model_path = Path(model_path)
    self.backend = backend
    self.precision = precision
    self.device_id = device_id

    self._model: Any = None
    self._is_loaded = False
    self._warmup_done = False

    # Input/output metadata
    self.input_shapes: dict[str, tuple] = {}
    self.output_shapes: dict[str, tuple] = {}

  def load(self) -> None:
    """Load the model for the configured backend."""
    if self._is_loaded:
      return

    if self.backend == Backend.TENSORRT:
      self._load_tensorrt()
    elif self.backend == Backend.TINYGRAD_CUDA:
      self._load_tinygrad_cuda()
    elif self.backend == Backend.TINYGRAD_CPU:
      self._load_tinygrad_cpu()
    else:
      self._load_cpu()

    self._is_loaded = True

  def _load_tensorrt(self) -> None:
    """Load model with TensorRT backend."""
    try:
      import tensorrt  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:
      raise ImportError("TensorRT not available. Install with: pip install tensorrt") from e

    # Check if we have a pre-built engine
    engine_path = self.model_path.with_suffix(".trt")
    if engine_path.exists():
      self._load_trt_engine(engine_path)
    else:
      # Build engine from ONNX
      self._build_trt_engine()

  def _load_trt_engine(self, engine_path: Path) -> None:
    """Load pre-built TensorRT engine."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
      self._model = runtime.deserialize_cuda_engine(f.read())

  def _build_trt_engine(self) -> None:
    """Build TensorRT engine from ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(self.model_path, "rb") as f:
      if not parser.parse(f.read()):
        for i in range(parser.num_errors):
          print(f"TensorRT ONNX parser error: {parser.get_error(i)}")
        raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Set precision
    if self.precision == Precision.FP16:
      config.set_flag(trt.BuilderFlag.FP16)
    elif self.precision == Precision.BF16:
      config.set_flag(trt.BuilderFlag.BF16)
    elif self.precision == Precision.FP8:
      config.set_flag(trt.BuilderFlag.FP8)

    # Build engine
    self._model = builder.build_serialized_network(network, config)
    if self._model is None:
      raise RuntimeError("Failed to build TensorRT engine")

  def _load_tinygrad_cuda(self) -> None:
    """Load model with tinygrad CUDA backend."""
    os.environ["CUDA"] = "1"
    os.environ["DEV"] = "CUDA"

    self._load_tinygrad_model()

  def _load_tinygrad_cpu(self) -> None:
    """Load model with tinygrad CPU backend."""
    os.environ["DEV"] = "CPU"

    self._load_tinygrad_model()

  def _load_tinygrad_model(self) -> None:
    """Load pickled tinygrad model."""
    pkl_path = self.model_path.with_suffix(".pkl")
    if pkl_path.exists():
      with open(pkl_path, "rb") as f:
        self._model = pickle.load(f)
    else:
      # Try tinygrad-specific naming
      tg_path = self.model_path.with_name(self.model_path.stem + "_tinygrad.pkl")
      if tg_path.exists():
        with open(tg_path, "rb") as f:
          self._model = pickle.load(f)
      else:
        raise FileNotFoundError(f"No pickled model found for {self.model_path}")

  def _load_cpu(self) -> None:
    """Load model for CPU inference."""
    # Use ONNX Runtime for CPU
    try:
      import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError as e:
      raise ImportError("onnxruntime not available. Install with: pip install onnxruntime") from e

    self._model = ort.InferenceSession(
      str(self.model_path),
      providers=["CPUExecutionProvider"],
    )

  def warmup(self, runs: int = 10, inputs: dict[str, np.ndarray] | None = None) -> float:
    """Run warmup iterations for stable performance.

    Args:
      runs: Number of warmup runs
      inputs: Optional inputs (random inputs used if not provided)

    Returns:
      Total warmup time in milliseconds
    """
    self.load()

    if inputs is None:
      inputs = self._create_dummy_inputs()

    start = time.perf_counter()
    for _ in range(runs):
      self._run_inference(inputs)
    warmup_time = (time.perf_counter() - start) * 1000

    self._warmup_done = True
    return warmup_time

  def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Run single inference.

    Args:
      inputs: Dictionary of input tensors

    Returns:
      Model output as numpy array
    """
    self.load()
    return self._run_inference(inputs)

  def _run_inference(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute inference on loaded model."""
    if self.backend == Backend.TENSORRT:
      return self._run_tensorrt(inputs)
    elif self.backend in (Backend.TINYGRAD_CUDA, Backend.TINYGRAD_CPU):
      return self._run_tinygrad(inputs)
    else:
      return self._run_onnxruntime(inputs)

  def _run_tensorrt(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Run inference with TensorRT."""
    # TensorRT execution context management
    # This is a simplified version - full implementation would use CUDA streams
    raise NotImplementedError("TensorRT inference requires cuda-python setup")

  def _run_tinygrad(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Run inference with tinygrad."""
    from tinygrad import Tensor

    # Convert inputs to tinygrad tensors
    tg_inputs = {k: Tensor(v) for k, v in inputs.items()}

    # Run model
    output = self._model(**tg_inputs)

    # Get numpy output
    return output.contiguous().realize().numpy()

  def _run_onnxruntime(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Run inference with ONNX Runtime."""
    input_names = [i.name for i in self._model.get_inputs()]
    output_names = [o.name for o in self._model.get_outputs()]

    # Map inputs by name
    feed = {name: inputs.get(name) for name in input_names if name in inputs}

    outputs = self._model.run(output_names, feed)
    return outputs[0]  # Return first output

  def _create_dummy_inputs(self) -> dict[str, np.ndarray]:
    """Create dummy inputs for warmup."""
    # Load metadata if available
    metadata_path = self.model_path.with_name(self.model_path.stem + "_metadata.pkl")
    if metadata_path.exists():
      with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
        self.input_shapes = metadata.get("input_shapes", {})

    if not self.input_shapes:
      # Default shapes for openpilot models
      if "vision" in str(self.model_path):
        self.input_shapes = {
          "input_img": (1, 12, 128, 256),
          "calib": (1, 3),
        }
      elif "policy" in str(self.model_path):
        self.input_shapes = {
          "input": (1, 512),
        }
      else:
        self.input_shapes = {"input": (1, 256)}

    return {name: np.random.randn(*shape).astype(np.float32) for name, shape in self.input_shapes.items()}

  def profile(
    self,
    inputs: dict[str, np.ndarray] | None = None,
    runs: int = 100,
    warmup_runs: int = 20,
  ) -> ProfilingResult:
    """Profile model inference performance.

    Args:
      inputs: Input tensors (random if not provided)
      runs: Number of timed runs
      warmup_runs: Number of warmup runs before timing

    Returns:
      ProfilingResult with detailed statistics
    """
    self.load()

    if inputs is None:
      inputs = self._create_dummy_inputs()

    # Warmup
    self.warmup(warmup_runs, inputs)

    # Timed runs
    timings = []
    for _ in range(runs):
      start = time.perf_counter()
      self._run_inference(inputs)
      elapsed = (time.perf_counter() - start) * 1000
      timings.append(elapsed)

    timings_arr = np.array(timings)

    return ProfilingResult(
      model_name=self.model_path.stem,
      backend=self.backend,
      precision=self.precision,
      runs=runs,
      warmup_runs=warmup_runs,
      mean_time_ms=float(np.mean(timings_arr)),
      std_time_ms=float(np.std(timings_arr)),
      min_time_ms=float(np.min(timings_arr)),
      max_time_ms=float(np.max(timings_arr)),
      p50_time_ms=float(np.percentile(timings_arr, 50)),
      p95_time_ms=float(np.percentile(timings_arr, 95)),
      p99_time_ms=float(np.percentile(timings_arr, 99)),
      mean_fps=1000.0 / float(np.mean(timings_arr)),
      timings_ms=timings,
    )


def get_recommended_backend() -> Backend:
  """Get recommended backend for current hardware."""
  # Check for TensorRT
  try:
    import tensorrt  # noqa: F401

    return Backend.TENSORRT
  except ImportError:
    pass

  # Check for NVIDIA GPU
  try:
    from openpilot.system.hardware.nvidia.gpu import is_nvidia_available

    if is_nvidia_available():
      return Backend.TINYGRAD_CUDA
  except ImportError:
    pass

  return Backend.CPU


def get_recommended_precision(backend: Backend) -> Precision:
  """Get recommended precision for backend."""
  if backend == Backend.TENSORRT:
    # TensorRT works best with FP16
    return Precision.FP16

  if backend == Backend.TINYGRAD_CUDA:
    # Check GPU compute capability
    try:
      from openpilot.system.hardware.nvidia.gpu import get_best_gpu

      gpu = get_best_gpu()
      if gpu:
        if gpu.supports_nvfp4:  # type: ignore[truthy-function]
          return Precision.FP4
        if gpu.supports_fp8:  # type: ignore[truthy-function]
          return Precision.FP8
        if gpu.supports_bf16:  # type: ignore[truthy-function]
          return Precision.BF16
        if gpu.supports_fp16:  # type: ignore[truthy-function]
          return Precision.FP16
    except ImportError:
      pass
    return Precision.FP32

  return Precision.FP32


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Model runner with precision selection")
  parser.add_argument("model", help="Path to model file (.onnx)")
  parser.add_argument("--backend", choices=["tensorrt", "tinygrad_cuda", "cpu"], default="tensorrt")
  parser.add_argument("--precision", choices=["fp32", "fp16", "bf16", "fp8"], default="fp16")
  parser.add_argument("--runs", type=int, default=100, help="Number of profiling runs")
  parser.add_argument("--warmup", type=int, default=20, help="Number of warmup runs")
  args = parser.parse_args()

  backend = Backend(args.backend)
  precision = Precision(args.precision)

  runner = ModelRunner(args.model, backend=backend, precision=precision)

  print(f"Profiling {args.model}")
  print(f"Backend: {backend.value}, Precision: {precision.value}")
  print(f"Runs: {args.runs}, Warmup: {args.warmup}")
  print("-" * 50)

  result = runner.profile(runs=args.runs, warmup_runs=args.warmup)

  print(f"Mean: {result.mean_time_ms:.2f}ms ({result.mean_fps:.1f} FPS)")
  print(f"Std:  {result.std_time_ms:.2f}ms")
  print(f"Min:  {result.min_time_ms:.2f}ms")
  print(f"Max:  {result.max_time_ms:.2f}ms")
  print(f"P50:  {result.p50_time_ms:.2f}ms")
  print(f"P95:  {result.p95_time_ms:.2f}ms")
  print(f"P99:  {result.p99_time_ms:.2f}ms")
