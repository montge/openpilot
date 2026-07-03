"""Model export utilities for deployment.

Supports:
- ONNX export with optimization
- TensorRT export (if available)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
  import torch
  import torch.nn as nn

  TORCH_AVAILABLE = True
except ImportError:
  TORCH_AVAILABLE = False


@dataclass
class ExportConfig:
  """Export configuration.

  Attributes:
    opset_version: ONNX opset version
    input_shape: Model input shape (B, C, H, W)
    dynamic_batch: Allow dynamic batch size
    simplify: Run ONNX simplifier
    fp16: Export as FP16 (for TensorRT)
  """

  opset_version: int = 17
  input_shape: tuple[int, ...] = (1, 3, 256, 512)
  dynamic_batch: bool = True
  simplify: bool = True
  fp16: bool = False


def export_onnx(
  model: nn.Module,
  output_path: str | Path,
  config: ExportConfig | None = None,
) -> Path:
  """Export model to ONNX format.

  Args:
    model: PyTorch model to export
    output_path: Path for ONNX file
    config: Export configuration

  Returns:
    Path to exported ONNX file
  """
  if not TORCH_AVAILABLE:
    raise ImportError("PyTorch required for ONNX export")

  config = config or ExportConfig()
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  model.eval()

  # Create dummy input
  dummy_input = torch.randn(*config.input_shape)

  # Dynamic axes for batch size
  dynamic_axes = None
  if config.dynamic_batch:
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

  # Export
  torch.onnx.export(
    model,
    dummy_input,
    str(output_path),
    opset_version=config.opset_version,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
  )

  # Simplify if requested
  if config.simplify:
    try:
      import onnx
      from onnxsim import simplify as onnx_simplify

      onnx_model = onnx.load(str(output_path))
      simplified, ok = onnx_simplify(onnx_model)
      if ok:
        onnx.save(simplified, str(output_path))
    except ImportError:
      pass  # onnxsim not installed, skip simplification

  return output_path


def export_tensorrt(
  onnx_path: str | Path,
  output_path: str | Path,
  config: ExportConfig | None = None,
) -> Path:
  """Export ONNX model to TensorRT engine.

  Args:
    onnx_path: Path to ONNX model
    output_path: Path for TensorRT engine
    config: Export configuration

  Returns:
    Path to TensorRT engine file
  """
  try:
    import tensorrt as trt  # type: ignore[import-not-found]
  except ImportError as err:
    raise ImportError("TensorRT required. Install with: pip install tensorrt") from err

  config = config or ExportConfig()
  onnx_path = Path(onnx_path)
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  logger = trt.Logger(trt.Logger.WARNING)
  builder = trt.Builder(logger)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, logger)

  # Parse ONNX
  with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
      for i in range(parser.num_errors):
        print(f"TensorRT parse error: {parser.get_error(i)}")
      raise RuntimeError("Failed to parse ONNX model")

  # Build config
  build_config = builder.create_builder_config()
  build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

  if config.fp16:
    build_config.set_flag(trt.BuilderFlag.FP16)

  # Dynamic batch size
  if config.dynamic_batch:
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    shape = list(config.input_shape)
    min_shape = [1] + shape[1:]
    opt_shape = shape
    max_shape = [max(8, shape[0])] + shape[1:]
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    build_config.add_optimization_profile(profile)

  # Build engine
  engine = builder.build_serialized_network(network, build_config)
  if engine is None:
    raise RuntimeError("Failed to build TensorRT engine")

  with open(output_path, "wb") as f:
    f.write(engine)

  return output_path
