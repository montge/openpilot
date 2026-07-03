"""Quantization utilities for FAIR student models.

Provides quantization-aware training, INT8 post-training quantization,
and model export to ONNX/TensorRT formats.
"""

from __future__ import annotations

try:
  from openpilot.tools.fair.quantization.qat import (
    QuantizationConfig,
    prepare_qat,
    convert_to_quantized,
  )
  from openpilot.tools.fair.quantization.export import (
    export_onnx,
    export_tensorrt,
    ExportConfig,
  )

  QAT_AVAILABLE = True
except ImportError:
  QAT_AVAILABLE = False

__all__ = [
  "QuantizationConfig",
  "prepare_qat",
  "convert_to_quantized",
  "export_onnx",
  "export_tensorrt",
  "ExportConfig",
  "QAT_AVAILABLE",
]
