"""Read the I/O contract embedded in an openpilot ONNX model.

Upstream deleted openpilot.selfdrive.modeld.get_model_metadata when modeld moved to
tinygrad's generic ONNX compiler (#38922): modeld now reads the contract out of its
compiled artifact. The DGX tooling works on the ONNX directly (TensorRT, onnxruntime,
training), so it keeps this fork-owned reader with the same make_metadata_dict() shape.

The driving model is recurrent since #38916: every `state_*` input has a matching
`next_state_*` output that must be fed back on the next step (see `state_pairs`).
"""

import codecs
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad.dtype import _to_np_dtype
from tinygrad.nn.onnx import OnnxPBParser


class MetadataOnnxPBParser(OnnxPBParser):
  """Parses only the graph and metadata_props of a ModelProto."""

  def _parse_ModelProto(self) -> dict:
    obj: dict[str, Any] = {"graph": {"input": [], "output": []}, "metadata_props": []}
    for fid, wire_type in self._parse_message(self.reader.len):
      match fid:
        case 7:
          obj["graph"] = self._parse_GraphProto()
        case 14:
          obj["metadata_props"].append(self._parse_StringStringEntryProto())
        case _:
          self.reader.skip_field(wire_type)
    return obj


def _shape(value_info: dict[str, Any]) -> tuple[int, ...]:
  return tuple(int(dim) if isinstance(dim, int) else 0 for dim in value_info["parsed_type"].shape)


def make_metadata_dict(model_path: str | Path) -> dict[str, Any]:
  model = MetadataOnnxPBParser(Path(model_path)).parse()
  props = {p["key"]: p["value"] for p in model["metadata_props"]}
  assert "output_slices" in props, "output_slices not found in metadata"
  inputs, outputs = model["graph"]["input"], model["graph"]["output"]
  input_shapes = {x["name"]: _shape(x) for x in inputs}
  output_shapes = {x["name"]: _shape(x) for x in outputs}
  return {
    "model_checkpoint": props.get("model_checkpoint"),
    "output_slices": pickle.loads(codecs.decode(props["output_slices"].encode(), "base64")),
    "input_shapes": input_shapes,
    "input_dtypes": {x["name"]: np.dtype(_to_np_dtype(x["parsed_type"].dtype)) for x in inputs},
    "output_shapes": output_shapes,
    "state_pairs": {name: f"next_{name}" for name in input_shapes if f"next_{name}" in output_shapes},
  }
