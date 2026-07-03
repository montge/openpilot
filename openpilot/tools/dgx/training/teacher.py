"""TensorRT-accelerated teacher model for pseudo-label generation.

Uses TensorRT for 800+ FPS inference, enabling fast pseudo-label
generation for knowledge distillation training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openpilot.selfdrive.modeld.constants import Plan
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.selfdrive.modeld.parse_model_outputs import Parser

# Type hints for optional imports
try:
  import tensorrt as trt  # type: ignore[import-not-found]

  TRT_AVAILABLE = True
except ImportError:
  TRT_AVAILABLE = False
  trt = None


class TensorRTEngine:
  """TensorRT engine wrapper for fast inference."""

  def __init__(self, onnx_path: str, fp16: bool = True, verbose: bool = False):
    if not TRT_AVAILABLE:
      raise RuntimeError("TensorRT not installed. Run: pip install tensorrt")

    self.onnx_path = onnx_path
    self.fp16 = fp16
    self.verbose = verbose

    # Build or load engine
    self.engine = self._build_engine()
    self.context = self.engine.create_execution_context()

    # Setup I/O buffers
    self._setup_buffers()

  def _build_engine(self):
    """Build TensorRT engine from ONNX model."""
    logger = trt.Logger(trt.Logger.WARNING if self.verbose else trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(self.onnx_path, "rb") as f:
      if not parser.parse(f.read()):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"Failed to parse ONNX: {errors}")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    if self.fp16:
      config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    serialized = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(serialized)

  def _setup_buffers(self):
    """Setup input/output buffers."""
    self.inputs = {}
    self.outputs = {}
    self.input_shapes = {}
    self.output_shapes = {}

    for i in range(self.engine.num_io_tensors):
      name = self.engine.get_tensor_name(i)
      shape = tuple(self.engine.get_tensor_shape(name))
      dtype = trt.nptype(self.engine.get_tensor_dtype(name))

      if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        self.input_shapes[name] = shape
        self.inputs[name] = np.zeros(shape, dtype=dtype)
      else:
        self.output_shapes[name] = shape
        self.outputs[name] = np.zeros(shape, dtype=dtype)

  def __call__(self, **inputs) -> dict[str, np.ndarray]:
    """Run inference.

    Args:
      **inputs: Input tensors as keyword arguments

    Returns:
      Dictionary of output tensors
    """
    # Copy inputs
    for name, data in inputs.items():
      if name in self.inputs:
        np.copyto(self.inputs[name], data)
        self.context.set_tensor_address(name, self.inputs[name].ctypes.data)

    # Set output addresses
    for name, buf in self.outputs.items():
      self.context.set_tensor_address(name, buf.ctypes.data)

    # Execute
    self.context.execute_async_v3(0)

    # Return copies of outputs
    return {name: buf.copy() for name, buf in self.outputs.items()}


def parse_supercombo_outputs(raw_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
  """Decompose and parse the flat supercombo output vector.

  Mirrors modeld's slice_outputs + Parser().parse_outputs flow
  (openpilot/selfdrive/modeld/modeld.py), batched.

  Args:
    raw_outputs: (batch, N) flat combined model output
    output_slices: name -> slice mapping from the ONNX-embedded metadata

  Returns:
    Parsed outputs keyed by name (plan, plan_stds, lane_lines, road_edges,
    lead, meta, desire_state, hidden_state, ...). Values keep the batch dim.
  """
  # Parser mutates in place (softmax/exp on views), so slice copies
  sliced = {k: raw_outputs[:, v].copy() for k, v in output_slices.items()}
  return Parser().parse_outputs(sliced)


class TeacherModel:
  """Combined supercombo teacher for pseudo-label generation.

  Runs the single driving_supercombo.onnx (merged vision+policy graph);
  input/output specs come from the metadata embedded in the ONNX.
  """

  def __init__(
    self,
    models_dir: str = "openpilot/selfdrive/modeld/models",
    model_name: str = "driving_supercombo.onnx",
    fp16: bool = True,
    verbose: bool = False,
  ):
    self.models_dir = Path(models_dir)
    self.fp16 = fp16
    self.verbose = verbose

    model_path = self.models_dir / model_name
    if not model_path.exists():
      raise FileNotFoundError(f"Model not found: {model_path}")

    self.metadata = make_metadata_dict(model_path)
    self.input_shapes: dict[str, tuple[int, ...]] = self.metadata["input_shapes"]
    self.output_slices: dict[str, slice] = self.metadata["output_slices"]

    print(f"Loading teacher model with TensorRT...\n  Building {model_name}...")
    self.model = TensorRTEngine(str(model_path), fp16=self.fp16, verbose=self.verbose)
    print("Teacher model loaded!")

  def generate_labels(
    self,
    img: np.ndarray,
    big_img: np.ndarray,
    desire: np.ndarray,
    traffic_convention: np.ndarray,
    features_buffer: np.ndarray | None = None,
    action_t: np.ndarray | None = None,
  ) -> dict[str, np.ndarray]:
    """Generate pseudo-labels for a batch of frames.

    The engine is built for batch 1 (the ONNX has fixed shapes), so frames
    are run one at a time. Without an explicit features_buffer the recurrent
    feature history is zero (cold start) — fine for shuffled training frames,
    but sequential streams get better labels by threading each frame's
    returned features back in.

    Args:
      img: (batch, 12, 128, 256) uint8 camera frames (2 stacked YUV frames)
      big_img: (batch, 12, 128, 256) uint8 wide camera frames
      desire: (batch, 8) desire vector, placed in the last desire_pulse step
      traffic_convention: (batch, 2) traffic convention
      features_buffer: (batch, 24, 512) prior feature history, zeros if None
      action_t: (batch, 2) previous action, zeros if None

    Returns:
      Dictionary with:
        - features: (batch, 512) hidden_state features (recurrent feedback)
        - raw_outputs: (batch, N) flat model output
        - path_mean: (batch, 1, 33, 3) plan position mean (single hypothesis)
        - path_std: (batch, 1, 33, 3) plan position std
        - path_prob: (batch, 1) hypothesis probability (always 1)
        - all parsed outputs (plan, plan_stds, lane_lines, road_edges, lead, ...)
    """
    batch_size = img.shape[0]
    pulse_shape = self.input_shapes["desire_pulse"][1:]  # (25, 8)
    feat_shape = self.input_shapes["features_buffer"][1:]  # (24, 512)

    if features_buffer is None:
      features_buffer = np.zeros((batch_size, *feat_shape), dtype=np.float32)
    if action_t is None:
      action_t = np.zeros((batch_size, self.input_shapes["action_t"][-1]), dtype=np.float32)

    raw = []
    for i in range(batch_size):
      desire_pulse = np.zeros((1, *pulse_shape), dtype=np.float32)
      desire_pulse[0, -1, :] = desire[i]

      out = self.model(
        img=img[i : i + 1],
        big_img=big_img[i : i + 1],
        desire_pulse=desire_pulse,
        traffic_convention=traffic_convention[i : i + 1],
        features_buffer=features_buffer[i : i + 1],
        action_t=action_t[i : i + 1],
      )
      raw.append(out.get("outputs", next(iter(out.values()))).reshape(1, -1))

    raw_outputs = np.concatenate(raw, axis=0).astype(np.float32)
    parsed = parse_supercombo_outputs(raw_outputs, self.output_slices)

    plan = parsed["plan"]  # (batch, 33, 15)
    plan_stds = parsed["plan_stds"]
    return {
      **parsed,
      "features": raw_outputs[:, self.output_slices["hidden_state"]],
      "raw_outputs": raw_outputs,
      "path_mean": plan[:, np.newaxis, :, Plan.POSITION],
      "path_std": plan_stds[:, np.newaxis, :, Plan.POSITION],
      "path_prob": np.ones((batch_size, 1), dtype=np.float32),
    }


def create_teacher(
  models_dir: str = "openpilot/selfdrive/modeld/models",
  fp16: bool = True,
) -> TeacherModel:
  """Factory function to create teacher model."""
  return TeacherModel(models_dir=models_dir, fp16=fp16)
