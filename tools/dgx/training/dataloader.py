"""Route log dataloader for training openpilot models.

Loads driving data from route logs and prepares it for training.
Supports loading from:
  - Local route logs
  - commaCarSegments (HuggingFace)
  - CI test segments (openpilotci blob storage)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  import torch  # type: ignore[import-not-found]

# These imports are available in openpilot environment
try:
  from openpilot.tools.lib.logreader import LogReader
  from openpilot.tools.lib.route import Route, SegmentName
  from openpilot.tools.lib.comma_car_segments import get_comma_car_segments_database
except ImportError:
  LogReader = None  # type: ignore[misc, assignment]
  Route = None  # type: ignore[misc, assignment]
  SegmentName = None  # type: ignore[misc, assignment]


# CI test segments for quick validation
CI_TEST_SEGMENTS = [
  "0982d79ebb0de295|2021-01-04--17-13-21--13",  # TOYOTA.TOYOTA_PRIUS
  "0982d79ebb0de295|2021-01-03--20-03-36--6",  # TOYOTA.TOYOTA_RAV4
  "eb140f119469d9ab|2021-06-12--10-46-24--27",  # HONDA.HONDA_CIVIC
]

CI_BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci"


@dataclass
class TrainingSample:
  """Single training sample extracted from route logs."""

  # Camera inputs (uint8, shape depends on camera)
  road_frame: np.ndarray  # (6, 128, 256) - stacked temporal frames
  wide_frame: np.ndarray  # (6, 128, 256) - stacked temporal frames

  # Context inputs
  desire: np.ndarray  # (8,) one-hot desire vector
  traffic_convention: np.ndarray  # (2,) left/right hand drive

  # Labels from recorded model outputs
  model_outputs: np.ndarray | None  # (1000,) raw model output for distillation

  # Metadata
  timestamp: int
  segment_id: str


class RouteLogDataset:
  """Dataset that loads samples from route logs."""

  def __init__(
    self,
    segments: list[str],
    frame_stack: int = 6,
    stride: int = 1,
    use_cached: bool = True,
    cache_dir: str | None = None,
  ):
    """Initialize dataset.

    Args:
      segments: List of segment identifiers (e.g., "dongle_id|timestamp--seg_num")
      frame_stack: Number of frames to stack temporally
      stride: Sample every Nth frame
      use_cached: Whether to use cached preprocessed data
      cache_dir: Directory for cached data
    """
    self.segments = segments
    self.frame_stack = frame_stack
    self.stride = stride
    self.use_cached = use_cached
    self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/training_cache")

    if use_cached:
      self.cache_dir.mkdir(parents=True, exist_ok=True)

    # Build index of all samples across segments
    self._samples: list[tuple[str, int]] = []  # (segment_id, frame_idx)
    self._build_index()

  def _build_index(self):
    """Build index of valid samples from all segments."""
    for seg_id in self.segments:
      try:
        # Count frames in segment (approximate - 20 FPS * 60 sec = 1200 frames)
        # We'll validate during actual loading
        num_frames = 1200  # Approximate, actual count loaded lazily
        valid_start = self.frame_stack - 1
        for frame_idx in range(valid_start, num_frames, self.stride):
          self._samples.append((seg_id, frame_idx))
      except Exception as e:
        print(f"Warning: Could not index segment {seg_id}: {e}")

  def __len__(self) -> int:
    return len(self._samples)

  def __getitem__(self, idx: int) -> TrainingSample:
    """Get a training sample by index."""
    seg_id, frame_idx = self._samples[idx]
    return self._load_sample(seg_id, frame_idx)

  def _load_sample(self, seg_id: str, frame_idx: int) -> TrainingSample:
    """Load a single sample from route logs."""
    if LogReader is None:
      raise RuntimeError("openpilot tools not available. Run from openpilot environment.")

    # Load the log for this segment
    lr = LogReader(seg_id)

    # Collect relevant messages
    model_outputs = []
    desires = []
    traffic_conv = None

    for msg in lr:
      msg_type = msg.which()

      if msg_type == "roadCameraState":
        # Frame metadata - actual pixels need video decode
        pass
      elif msg_type == "wideRoadCameraState":
        pass
      elif msg_type == "modelV2":
        # Model predictions - these are our distillation targets
        model_data = msg.modelV2
        model_outputs.append(
          {
            "timestamp": msg.logMonoTime,
            "raw": self._extract_model_output(model_data),
          }
        )
      elif msg_type == "selfdriveState":
        # Desire state for lane changes
        state = msg.selfdriveState
        desires.append(
          {
            "timestamp": msg.logMonoTime,
            "desire": self._encode_desire(state),
          }
        )
      elif msg_type == "carState" and traffic_conv is None:
        # Get traffic convention from car state
        # Left-hand drive: [1, 0], Right-hand drive: [0, 1]
        traffic_conv = np.array([1.0, 0.0], dtype=np.float16)  # Default to left-hand

    # Find closest model output to frame_idx
    # For now, return placeholder - actual implementation needs video decoding
    return TrainingSample(
      road_frame=np.zeros((self.frame_stack, 128, 256), dtype=np.uint8),
      wide_frame=np.zeros((self.frame_stack, 128, 256), dtype=np.uint8),
      desire=np.zeros(8, dtype=np.float16),
      traffic_convention=traffic_conv if traffic_conv is not None else np.array([1.0, 0.0], dtype=np.float16),
      model_outputs=model_outputs[0]["raw"] if model_outputs else None,
      timestamp=0,
      segment_id=seg_id,
    )

  def _extract_model_output(self, model_data) -> np.ndarray:
    """Extract raw model output from modelV2 message."""
    # The modelV2 message contains parsed predictions
    # For distillation, we want the raw output vector
    # This is a simplified version - full implementation needs output format details
    outputs = []

    # Path predictions
    if hasattr(model_data, "position") and model_data.position:
      pos = model_data.position
      outputs.extend([pos.x, pos.y, pos.z])

    # Lane lines
    if hasattr(model_data, "laneLines"):
      for lane in model_data.laneLines:
        if hasattr(lane, "y"):
          outputs.extend(list(lane.y)[:33])  # First 33 points

    # Pad to expected size
    raw = np.zeros(1000, dtype=np.float32)
    raw[: len(outputs)] = outputs[:1000]
    return raw

  def _encode_desire(self, state) -> np.ndarray:
    """Encode desire state as one-hot vector."""
    # Desire values: NONE, TURN_LEFT, TURN_RIGHT, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, KEEP_LEFT, KEEP_RIGHT, ...
    desire = np.zeros(8, dtype=np.float16)
    if hasattr(state, "desireState"):
      # TODO: Implement proper desire encoding based on selfdriveState schema
      pass
    return desire


class CommaCarSegmentsDataset(RouteLogDataset):
  """Dataset loading from commaCarSegments on HuggingFace."""

  def __init__(
    self,
    platforms: list[str] | None = None,
    max_segments_per_platform: int | None = None,
    **kwargs,
  ):
    """Initialize commaCarSegments dataset.

    Args:
      platforms: List of car platforms to include (None = all)
      max_segments_per_platform: Limit segments per platform
      **kwargs: Additional args passed to RouteLogDataset
    """
    # Get segment database
    if get_comma_car_segments_database is None:
      raise RuntimeError("commaCarSegments not available")

    database = get_comma_car_segments_database()

    segments = []
    for platform, platform_segments in database.items():
      if platforms is not None and platform not in platforms:
        continue

      if max_segments_per_platform:
        platform_segments = platform_segments[:max_segments_per_platform]

      segments.extend(platform_segments)

    print(f"Found {len(segments)} segments across {len(database)} platforms")
    super().__init__(segments=segments, **kwargs)


class CITestDataset(RouteLogDataset):
  """Dataset using CI test segments for quick validation."""

  def __init__(self, **kwargs):
    """Initialize with CI test segments."""
    super().__init__(segments=CI_TEST_SEGMENTS, **kwargs)


def create_dataloader(
  dataset: RouteLogDataset,
  batch_size: int = 32,
  shuffle: bool = True,
  num_workers: int = 4,
  pin_memory: bool = True,
) -> torch.utils.data.DataLoader:  # type: ignore[name-defined]
  """Create PyTorch DataLoader from dataset.

  Args:
    dataset: RouteLogDataset instance
    batch_size: Batch size
    shuffle: Whether to shuffle
    num_workers: Number of data loading workers
    pin_memory: Pin memory for GPU transfer

  Returns:
    PyTorch DataLoader
  """
  import torch  # type: ignore[import-not-found]
  from torch.utils.data import DataLoader  # type: ignore[import-not-found]

  def collate_fn(samples: list[TrainingSample]) -> dict:
    """Collate samples into batched tensors."""
    return {
      "road_frame": torch.from_numpy(np.stack([s.road_frame for s in samples])),
      "wide_frame": torch.from_numpy(np.stack([s.wide_frame for s in samples])),
      "desire": torch.from_numpy(np.stack([s.desire for s in samples])),
      "traffic_convention": torch.from_numpy(np.stack([s.traffic_convention for s in samples])),
      "model_outputs": torch.from_numpy(np.stack([s.model_outputs for s in samples if s.model_outputs is not None])),
      "segment_ids": [s.segment_id for s in samples],
    }

  return DataLoader(
    dataset,  # type: ignore[arg-type]
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=collate_fn,
  )


def download_ci_segment(segment_id: str, output_dir: str | Path) -> Path:
  """Download a CI test segment for local testing.

  Args:
    segment_id: Segment identifier (e.g., "dongle_id|timestamp--seg_num")
    output_dir: Directory to save the segment

  Returns:
    Path to downloaded rlog file
  """
  import requests

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Parse segment ID: "dongle_id|timestamp--seg_num" e.g. "0982d79ebb0de295|2021-01-04--17-13-21--13"
  dongle_id, rest = segment_id.split("|")
  timestamp, seg_num = rest.rsplit("--", 1)

  # Build URL
  url = f"{CI_BASE_URL}/{dongle_id}/{timestamp}/{seg_num}/rlog.zst"

  # Download
  output_path = output_dir / f"{segment_id.replace('|', '_').replace('--', '_')}_rlog.zst"

  if output_path.exists():
    print(f"Already downloaded: {output_path}")
    return output_path

  print(f"Downloading {url}...")
  resp = requests.get(url, stream=True)
  resp.raise_for_status()

  with open(output_path, "wb") as f:
    for chunk in resp.iter_content(chunk_size=8192):
      f.write(chunk)

  print(f"Downloaded to: {output_path}")
  return output_path


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Test route log dataloader")
  parser.add_argument("--download-ci", action="store_true", help="Download CI test segments")
  parser.add_argument("--output-dir", type=str, default="/tmp/openpilot_segments", help="Output directory")
  args = parser.parse_args()

  if args.download_ci:
    print("Downloading CI test segments...")
    for seg in CI_TEST_SEGMENTS[:1]:  # Just first segment for testing
      download_ci_segment(seg, args.output_dir)
  else:
    print("Creating dataset from CI test segments...")
    try:
      dataset = CITestDataset()
      print(f"Dataset size: {len(dataset)}")
      if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample: {sample.segment_id}, frame shape: {sample.road_frame.shape}")
    except Exception as e:
      print(f"Error creating dataset: {e}")
      print("Run with --download-ci first to download test data")
