"""Log alignment for shadow device comparison.

Synchronizes logs from shadow and production devices using GPS time
or frame ID matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.tools.shadow.comparison_logger import FrameData


@dataclass
class AlignedPair:
  """A pair of aligned frames from shadow and production devices."""

  shadow_frame: FrameData
  production_frame: FrameData
  time_offset_ms: float  # Estimated time offset between devices
  alignment_quality: float  # 0-1 quality score


@dataclass
class AlignmentResult:
  """Result of log alignment operation."""

  pairs: list[AlignedPair]
  shadow_only: list[FrameData]  # Frames only in shadow log
  production_only: list[FrameData]  # Frames only in production log
  mean_time_offset_ms: float
  alignment_quality: float  # Overall quality score 0-1
  method: str  # "gps", "frame_id", or "timestamp"


class LogAligner:
  """Aligns logs from shadow and production devices.

  Supports multiple alignment methods:
  1. GPS time - Uses GPS timestamps for precise alignment (best accuracy)
  2. Frame ID - Uses frame IDs if devices are synchronized
  3. Timestamp interpolation - Uses monotonic timestamps with offset estimation

  Usage:
    aligner = LogAligner()
    result = aligner.align_by_gps(shadow_frames, production_frames)

    for pair in result.pairs:
      shadow_steer = pair.shadow_frame.controls["steer"]
      prod_steer = pair.production_frame.controls["steer"]
      delta = abs(shadow_steer - prod_steer)
  """

  # Tolerance for GPS timestamp matching (seconds)
  GPS_TOLERANCE_S = 0.05  # 50ms

  # Tolerance for frame ID matching
  FRAME_ID_TOLERANCE = 5

  # Minimum quality score for valid alignment
  MIN_QUALITY = 0.5

  def __init__(
    self,
    gps_tolerance_s: float | None = None,
    frame_id_tolerance: int | None = None,
  ):
    """Initialize the aligner.

    Args:
      gps_tolerance_s: Override for GPS timestamp tolerance
      frame_id_tolerance: Override for frame ID tolerance
    """
    self.gps_tolerance_s = gps_tolerance_s or self.GPS_TOLERANCE_S
    self.frame_id_tolerance = frame_id_tolerance or self.FRAME_ID_TOLERANCE

  def align_by_gps(
    self,
    shadow_frames: list[FrameData],
    production_frames: list[FrameData],
  ) -> AlignmentResult:
    """Align logs using GPS timestamps.

    This is the most accurate method when both devices have GPS.

    Args:
      shadow_frames: Frames from shadow device
      production_frames: Frames from production device

    Returns:
      AlignmentResult with paired frames
    """
    pairs: list[AlignedPair] = []
    shadow_matched: set[int] = set()
    production_matched: set[int] = set()

    # Filter frames with GPS timestamps
    shadow_gps = [(i, f) for i, f in enumerate(shadow_frames) if f.timestamp_gps is not None]
    prod_gps = [(i, f) for i, f in enumerate(production_frames) if f.timestamp_gps is not None]

    if not shadow_gps or not prod_gps:
      # Fall back to timestamp alignment if no GPS
      return self.align_by_timestamp(shadow_frames, production_frames)

    # Build index of production frames by GPS time (timestamp_gps is guaranteed non-None by filter above)
    prod_by_time: list[tuple[float, int, FrameData]] = [
      (f.timestamp_gps, i, f)  # type: ignore[misc]
      for i, f in prod_gps
    ]
    prod_by_time.sort(key=lambda x: x[0])
    prod_times = np.array([t for t, _, _ in prod_by_time])

    time_offsets: list[float] = []

    for shadow_idx, shadow_frame in shadow_gps:
      shadow_gps_time = shadow_frame.timestamp_gps
      if shadow_gps_time is None:
        continue

      # Find closest production frame
      closest_idx = np.searchsorted(prod_times, shadow_gps_time)

      # Check neighbors
      candidates = []
      for check_idx in [closest_idx - 1, closest_idx, closest_idx + 1]:
        if 0 <= check_idx < len(prod_by_time):
          prod_time, prod_orig_idx, prod_frame = prod_by_time[check_idx]
          time_diff = abs(prod_time - shadow_gps_time)
          if time_diff <= self.gps_tolerance_s:
            candidates.append((time_diff, prod_orig_idx, prod_frame))

      if candidates:
        # Pick closest
        candidates.sort(key=lambda x: x[0])
        time_diff, prod_orig_idx, prod_frame = candidates[0]

        if prod_orig_idx not in production_matched:
          # Quality based on time difference (closer = higher quality)
          quality = max(0.0, 1.0 - (time_diff / self.gps_tolerance_s))
          time_offset_ms = (shadow_gps_time - prod_frame.timestamp_gps) * 1000  # type: ignore[operator]

          pairs.append(
            AlignedPair(
              shadow_frame=shadow_frame,
              production_frame=prod_frame,
              time_offset_ms=time_offset_ms,
              alignment_quality=quality,
            )
          )

          shadow_matched.add(shadow_idx)
          production_matched.add(prod_orig_idx)
          time_offsets.append(time_offset_ms)

    # Collect unmatched frames
    shadow_only = [f for i, f in enumerate(shadow_frames) if i not in shadow_matched]
    production_only = [f for i, f in enumerate(production_frames) if i not in production_matched]

    # Compute overall quality
    if pairs:
      mean_quality = sum(p.alignment_quality for p in pairs) / len(pairs)
      mean_offset = float(np.mean(time_offsets)) if time_offsets else 0.0
    else:
      mean_quality = 0.0
      mean_offset = 0.0

    return AlignmentResult(
      pairs=pairs,
      shadow_only=shadow_only,
      production_only=production_only,
      mean_time_offset_ms=mean_offset,
      alignment_quality=mean_quality,
      method="gps",
    )

  def align_by_frame_id(
    self,
    shadow_frames: list[FrameData],
    production_frames: list[FrameData],
  ) -> AlignmentResult:
    """Align logs using frame IDs.

    Works when both devices use synchronized frame counters.

    Args:
      shadow_frames: Frames from shadow device
      production_frames: Frames from production device

    Returns:
      AlignmentResult with paired frames
    """
    pairs: list[AlignedPair] = []
    shadow_matched: set[int] = set()
    production_matched: set[int] = set()

    # Build index of production frames by frame_id
    prod_by_id: dict[int, tuple[int, FrameData]] = {f.frame_id: (i, f) for i, f in enumerate(production_frames)}

    time_offsets: list[float] = []

    for shadow_idx, shadow_frame in enumerate(shadow_frames):
      shadow_id = shadow_frame.frame_id

      # Check for exact match or within tolerance
      for delta in range(self.frame_id_tolerance + 1):
        for candidate_id in [shadow_id + delta, shadow_id - delta]:
          if candidate_id in prod_by_id:
            prod_idx, prod_frame = prod_by_id[candidate_id]

            if prod_idx not in production_matched:
              # Quality based on frame ID difference
              quality = 1.0 - (abs(delta) / (self.frame_id_tolerance + 1))

              # Time offset from monotonic timestamps
              time_offset_ms = (shadow_frame.timestamp_mono - prod_frame.timestamp_mono) * 1000

              pairs.append(
                AlignedPair(
                  shadow_frame=shadow_frame,
                  production_frame=prod_frame,
                  time_offset_ms=time_offset_ms,
                  alignment_quality=quality,
                )
              )

              shadow_matched.add(shadow_idx)
              production_matched.add(prod_idx)
              time_offsets.append(time_offset_ms)
              break
        else:
          continue
        break

    # Collect unmatched frames
    shadow_only = [f for i, f in enumerate(shadow_frames) if i not in shadow_matched]
    production_only = [f for i, f in enumerate(production_frames) if i not in production_matched]

    # Compute overall quality
    if pairs:
      mean_quality = sum(p.alignment_quality for p in pairs) / len(pairs)
      mean_offset = float(np.mean(time_offsets)) if time_offsets else 0.0
    else:
      mean_quality = 0.0
      mean_offset = 0.0

    return AlignmentResult(
      pairs=pairs,
      shadow_only=shadow_only,
      production_only=production_only,
      mean_time_offset_ms=mean_offset,
      alignment_quality=mean_quality,
      method="frame_id",
    )

  def align_by_timestamp(
    self,
    shadow_frames: list[FrameData],
    production_frames: list[FrameData],
    estimate_offset: bool = True,
  ) -> AlignmentResult:
    """Align logs using monotonic timestamps with offset estimation.

    Uses cross-correlation to estimate the time offset between devices,
    then pairs frames with similar adjusted timestamps.

    Args:
      shadow_frames: Frames from shadow device
      production_frames: Frames from production device
      estimate_offset: Whether to estimate time offset first

    Returns:
      AlignmentResult with paired frames
    """
    if not shadow_frames or not production_frames:
      return AlignmentResult(
        pairs=[],
        shadow_only=shadow_frames,
        production_only=production_frames,
        mean_time_offset_ms=0.0,
        alignment_quality=0.0,
        method="timestamp",
      )

    # Estimate offset using first timestamps
    shadow_start = shadow_frames[0].timestamp_mono
    prod_start = production_frames[0].timestamp_mono

    # Simple offset: assume recordings started at same real time
    estimated_offset = shadow_start - prod_start if estimate_offset else 0.0

    pairs: list[AlignedPair] = []
    shadow_matched: set[int] = set()
    production_matched: set[int] = set()

    # Tolerance for timestamp matching (in seconds)
    timestamp_tolerance = 0.05  # 50ms

    # Build sorted list of production timestamps
    prod_times = np.array([f.timestamp_mono for f in production_frames])

    for shadow_idx, shadow_frame in enumerate(shadow_frames):
      # Adjust shadow timestamp by estimated offset
      adjusted_time = shadow_frame.timestamp_mono - estimated_offset

      # Find closest production frame
      closest_idx = int(np.searchsorted(prod_times, adjusted_time))

      # Check neighbors
      best_match = None
      best_diff = float("inf")

      for check_idx in [closest_idx - 1, closest_idx, closest_idx + 1]:
        if 0 <= check_idx < len(production_frames):
          prod_frame = production_frames[check_idx]
          time_diff = abs(prod_frame.timestamp_mono - adjusted_time)

          if time_diff < best_diff and time_diff <= timestamp_tolerance:
            if check_idx not in production_matched:
              best_match = (check_idx, prod_frame, time_diff)
              best_diff = time_diff

      if best_match:
        prod_idx, prod_frame, time_diff = best_match

        # Quality based on time difference
        quality = max(0.0, 1.0 - (time_diff / timestamp_tolerance))
        time_offset_ms = (shadow_frame.timestamp_mono - prod_frame.timestamp_mono) * 1000

        pairs.append(
          AlignedPair(
            shadow_frame=shadow_frame,
            production_frame=prod_frame,
            time_offset_ms=time_offset_ms,
            alignment_quality=quality,
          )
        )

        shadow_matched.add(shadow_idx)
        production_matched.add(prod_idx)

    # Collect unmatched frames
    shadow_only = [f for i, f in enumerate(shadow_frames) if i not in shadow_matched]
    production_only = [f for i, f in enumerate(production_frames) if i not in production_matched]

    # Compute overall quality
    if pairs:
      mean_quality = sum(p.alignment_quality for p in pairs) / len(pairs)
      mean_offset = sum(p.time_offset_ms for p in pairs) / len(pairs)
    else:
      mean_quality = 0.0
      mean_offset = 0.0

    return AlignmentResult(
      pairs=pairs,
      shadow_only=shadow_only,
      production_only=production_only,
      mean_time_offset_ms=mean_offset,
      alignment_quality=mean_quality,
      method="timestamp",
    )

  def auto_align(
    self,
    shadow_frames: list[FrameData],
    production_frames: list[FrameData],
  ) -> AlignmentResult:
    """Automatically choose best alignment method.

    Tries GPS first, then frame ID, then timestamp.

    Args:
      shadow_frames: Frames from shadow device
      production_frames: Frames from production device

    Returns:
      AlignmentResult with paired frames
    """
    # Check if GPS timestamps available
    shadow_has_gps = any(f.timestamp_gps is not None for f in shadow_frames)
    prod_has_gps = any(f.timestamp_gps is not None for f in production_frames)

    if shadow_has_gps and prod_has_gps:
      result = self.align_by_gps(shadow_frames, production_frames)
      if result.alignment_quality >= self.MIN_QUALITY:
        return result

    # Try frame ID matching
    result = self.align_by_frame_id(shadow_frames, production_frames)
    if result.alignment_quality >= self.MIN_QUALITY:
      return result

    # Fall back to timestamp
    return self.align_by_timestamp(shadow_frames, production_frames)


def merge_aligned_logs(
  result: AlignmentResult,
  include_unmatched: bool = False,
) -> list[dict[str, Any]]:
  """Merge aligned logs into a single list for analysis.

  Args:
    result: AlignmentResult from alignment
    include_unmatched: Whether to include unmatched frames

  Returns:
    List of merged frame dictionaries with shadow_ and prod_ prefixes
  """
  merged: list[dict[str, Any]] = []

  for pair in result.pairs:
    entry = {
      "frame_id": pair.shadow_frame.frame_id,
      "timestamp_mono": pair.shadow_frame.timestamp_mono,
      "timestamp_gps": pair.shadow_frame.timestamp_gps,
      "time_offset_ms": pair.time_offset_ms,
      "alignment_quality": pair.alignment_quality,
      "shadow_controls": pair.shadow_frame.controls,
      "prod_controls": pair.production_frame.controls,
      "shadow_model_outputs": pair.shadow_frame.model_outputs,
      "prod_model_outputs": pair.production_frame.model_outputs,
      "shadow_trajectory": pair.shadow_frame.trajectory,
      "prod_trajectory": pair.production_frame.trajectory,
      "shadow_state": pair.shadow_frame.state,
      "prod_state": pair.production_frame.state,
    }
    merged.append(entry)

  if include_unmatched:
    for frame in result.shadow_only:
      unmatched_entry: dict[str, Any] = {
        "frame_id": frame.frame_id,
        "timestamp_mono": frame.timestamp_mono,
        "timestamp_gps": frame.timestamp_gps,
        "time_offset_ms": None,
        "alignment_quality": 0.0,
        "shadow_controls": frame.controls,
        "prod_controls": None,
        "shadow_model_outputs": frame.model_outputs,
        "prod_model_outputs": None,
        "shadow_trajectory": frame.trajectory,
        "prod_trajectory": None,
        "shadow_state": frame.state,
        "prod_state": None,
        "unmatched": "shadow",
      }
      merged.append(unmatched_entry)

    for frame in result.production_only:
      unmatched_entry = {
        "frame_id": frame.frame_id,
        "timestamp_mono": frame.timestamp_mono,
        "timestamp_gps": frame.timestamp_gps,
        "time_offset_ms": None,
        "alignment_quality": 0.0,
        "shadow_controls": None,
        "prod_controls": frame.controls,
        "shadow_model_outputs": None,
        "prod_model_outputs": frame.model_outputs,
        "shadow_trajectory": None,
        "prod_trajectory": frame.trajectory,
        "shadow_state": None,
        "prod_state": frame.state,
        "unmatched": "production",
      }
      merged.append(unmatched_entry)

  return merged


def validate_alignment(result: AlignmentResult) -> dict[str, Any]:
  """Validate alignment quality and return diagnostics.

  Args:
    result: AlignmentResult to validate

  Returns:
    Dictionary with validation metrics
  """
  if not result.pairs:
    return {
      "valid": False,
      "reason": "No pairs aligned",
      "pair_count": 0,
      "quality": 0.0,
    }

  # Check for consistent time offsets
  offsets = [p.time_offset_ms for p in result.pairs]
  offset_std = float(np.std(offsets)) if len(offsets) > 1 else 0.0

  # Check for quality
  qualities = [p.alignment_quality for p in result.pairs]
  min_quality = min(qualities)
  mean_quality = float(np.mean(qualities))

  # Check coverage
  total_frames = len(result.pairs) + len(result.shadow_only) + len(result.production_only)
  coverage = len(result.pairs) / total_frames if total_frames > 0 else 0.0

  # Determine validity
  valid = (
    mean_quality >= 0.5
    and offset_std < 100.0  # Less than 100ms standard deviation
    and coverage >= 0.5  # At least 50% of frames matched
  )

  return {
    "valid": valid,
    "pair_count": len(result.pairs),
    "shadow_only_count": len(result.shadow_only),
    "production_only_count": len(result.production_only),
    "quality": mean_quality,
    "min_quality": min_quality,
    "mean_time_offset_ms": result.mean_time_offset_ms,
    "time_offset_std_ms": offset_std,
    "coverage": coverage,
    "method": result.method,
  }
