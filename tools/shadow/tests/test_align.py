"""Unit tests for log alignment."""

from __future__ import annotations


from openpilot.tools.shadow.align import (
  AlignmentResult,
  LogAligner,
  merge_aligned_logs,
  validate_alignment,
)
from openpilot.tools.shadow.comparison_logger import FrameData


def make_frame(
  frame_id: int,
  timestamp_mono: float,
  timestamp_gps: float | None = None,
  controls: dict | None = None,
) -> FrameData:
  """Helper to create test frames."""
  return FrameData(
    frame_id=frame_id,
    timestamp_mono=timestamp_mono,
    timestamp_gps=timestamp_gps,
    controls=controls or {},
  )


class TestLogAlignerGPS:
  """Tests for GPS-based alignment."""

  def test_align_by_gps_exact_match(self):
    """Test GPS alignment with exact matching timestamps."""
    aligner = LogAligner()

    shadow = [
      make_frame(1, 100.0, 1700000001.0, {"steer": 0.1}),
      make_frame(2, 100.01, 1700000001.01, {"steer": 0.2}),
      make_frame(3, 100.02, 1700000001.02, {"steer": 0.3}),
    ]
    production = [
      make_frame(1, 200.0, 1700000001.0, {"steer": 0.11}),
      make_frame(2, 200.01, 1700000001.01, {"steer": 0.21}),
      make_frame(3, 200.02, 1700000001.02, {"steer": 0.31}),
    ]

    result = aligner.align_by_gps(shadow, production)

    assert len(result.pairs) == 3
    assert result.method == "gps"
    assert result.alignment_quality > 0.9

  def test_align_by_gps_with_offset(self):
    """Test GPS alignment with time offset between devices."""
    aligner = LogAligner()

    # Production device has slight GPS time offset (5ms)
    shadow = [
      make_frame(1, 100.0, 1700000001.0),
      make_frame(2, 100.01, 1700000001.1),  # 100ms later
    ]
    production = [
      make_frame(1, 200.0, 1700000001.005),  # 5ms offset
      make_frame(2, 200.01, 1700000001.105),  # 5ms offset
    ]

    result = aligner.align_by_gps(shadow, production)

    assert len(result.pairs) == 2
    assert abs(result.mean_time_offset_ms + 5) < 1  # ~-5ms offset

  def test_align_by_gps_no_gps_fallback(self):
    """Test fallback when no GPS timestamps available."""
    aligner = LogAligner()

    shadow = [make_frame(1, 100.0), make_frame(2, 100.01)]
    production = [make_frame(1, 100.0), make_frame(2, 100.01)]

    result = aligner.align_by_gps(shadow, production)

    # Should fall back to timestamp method
    assert result.method == "timestamp"

  def test_align_by_gps_unmatched_frames(self):
    """Test handling of unmatched frames."""
    aligner = LogAligner()

    shadow = [
      make_frame(1, 100.0, 1700000001.0),
      make_frame(2, 100.01, 1700000001.01),
      make_frame(3, 100.02, 1700000001.02),  # No match
    ]
    production = [
      make_frame(1, 200.0, 1700000001.0),
      make_frame(2, 200.01, 1700000001.01),
      make_frame(4, 200.03, 1700000002.0),  # Too far to match frame 3
    ]

    result = aligner.align_by_gps(shadow, production)

    assert len(result.pairs) == 2
    assert len(result.shadow_only) == 1
    assert len(result.production_only) == 1


class TestLogAlignerFrameID:
  """Tests for frame ID-based alignment."""

  def test_align_by_frame_id_exact(self):
    """Test frame ID alignment with exact IDs."""
    aligner = LogAligner()

    shadow = [
      make_frame(100, 1.0, controls={"steer": 0.1}),
      make_frame(101, 1.01, controls={"steer": 0.2}),
      make_frame(102, 1.02, controls={"steer": 0.3}),
    ]
    production = [
      make_frame(100, 2.0, controls={"steer": 0.11}),
      make_frame(101, 2.01, controls={"steer": 0.21}),
      make_frame(102, 2.02, controls={"steer": 0.31}),
    ]

    result = aligner.align_by_frame_id(shadow, production)

    assert len(result.pairs) == 3
    assert result.method == "frame_id"
    assert result.alignment_quality == 1.0  # Exact match

  def test_align_by_frame_id_with_offset(self):
    """Test frame ID alignment with ID offset within tolerance."""
    aligner = LogAligner(frame_id_tolerance=5)

    shadow = [make_frame(100, 1.0), make_frame(101, 1.01)]
    production = [make_frame(102, 2.0), make_frame(103, 2.01)]  # 2 ID offset

    result = aligner.align_by_frame_id(shadow, production)

    assert len(result.pairs) == 2
    assert result.alignment_quality < 1.0  # Not exact match

  def test_align_by_frame_id_beyond_tolerance(self):
    """Test frame ID alignment fails beyond tolerance."""
    aligner = LogAligner(frame_id_tolerance=2)

    shadow = [make_frame(100, 1.0)]
    production = [make_frame(110, 2.0)]  # 10 ID difference

    result = aligner.align_by_frame_id(shadow, production)

    assert len(result.pairs) == 0
    assert len(result.shadow_only) == 1
    assert len(result.production_only) == 1


class TestLogAlignerTimestamp:
  """Tests for timestamp-based alignment."""

  def test_align_by_timestamp_same_start(self):
    """Test timestamp alignment when recordings start at same time."""
    aligner = LogAligner()

    shadow = [
      make_frame(1, 100.0),
      make_frame(2, 100.01),
      make_frame(3, 100.02),
    ]
    production = [
      make_frame(1, 100.0),
      make_frame(2, 100.01),
      make_frame(3, 100.02),
    ]

    result = aligner.align_by_timestamp(shadow, production)

    assert len(result.pairs) == 3
    assert result.method == "timestamp"

  def test_align_by_timestamp_with_offset(self):
    """Test timestamp alignment with time offset."""
    aligner = LogAligner()

    # Production started 1 second later
    shadow = [make_frame(1, 100.0), make_frame(2, 100.01)]
    production = [make_frame(1, 101.0), make_frame(2, 101.01)]

    result = aligner.align_by_timestamp(shadow, production, estimate_offset=True)

    assert len(result.pairs) == 2

  def test_align_empty_lists(self):
    """Test alignment with empty frame lists."""
    aligner = LogAligner()

    result = aligner.align_by_timestamp([], [])

    assert len(result.pairs) == 0
    assert result.alignment_quality == 0.0


class TestAutoAlign:
  """Tests for automatic alignment method selection."""

  def test_auto_align_prefers_gps(self):
    """Test that auto_align prefers GPS when available."""
    aligner = LogAligner()

    shadow = [make_frame(1, 100.0, 1700000001.0)]
    production = [make_frame(1, 200.0, 1700000001.0)]

    result = aligner.auto_align(shadow, production)

    assert result.method == "gps"

  def test_auto_align_fallback_to_frame_id(self):
    """Test fallback to frame ID when GPS unavailable."""
    aligner = LogAligner()

    shadow = [make_frame(100, 1.0), make_frame(101, 1.01)]
    production = [make_frame(100, 2.0), make_frame(101, 2.01)]

    result = aligner.auto_align(shadow, production)

    assert result.method == "frame_id"


class TestMergeAlignedLogs:
  """Tests for log merging."""

  def test_merge_pairs_only(self):
    """Test merging aligned pairs."""
    shadow = make_frame(1, 100.0, controls={"steer": 0.1})
    prod = make_frame(1, 200.0, controls={"steer": 0.2})

    from openpilot.tools.shadow.align import AlignedPair

    result = AlignmentResult(
      pairs=[AlignedPair(shadow, prod, 0.0, 1.0)],
      shadow_only=[],
      production_only=[],
      mean_time_offset_ms=0.0,
      alignment_quality=1.0,
      method="gps",
    )

    merged = merge_aligned_logs(result)

    assert len(merged) == 1
    assert merged[0]["shadow_controls"]["steer"] == 0.1
    assert merged[0]["prod_controls"]["steer"] == 0.2

  def test_merge_include_unmatched(self):
    """Test merging with unmatched frames included."""

    shadow = make_frame(1, 100.0, controls={"steer": 0.1})
    prod = make_frame(2, 200.0, controls={"steer": 0.2})

    result = AlignmentResult(
      pairs=[],
      shadow_only=[shadow],
      production_only=[prod],
      mean_time_offset_ms=0.0,
      alignment_quality=0.0,
      method="gps",
    )

    merged = merge_aligned_logs(result, include_unmatched=True)

    assert len(merged) == 2
    assert merged[0]["unmatched"] == "shadow"
    assert merged[1]["unmatched"] == "production"


class TestValidateAlignment:
  """Tests for alignment validation."""

  def test_validate_good_alignment(self):
    """Test validation of high-quality alignment."""
    from openpilot.tools.shadow.align import AlignedPair

    pairs = [
      AlignedPair(
        make_frame(i, float(i)),
        make_frame(i, float(i)),
        0.0,
        0.95,
      )
      for i in range(10)
    ]

    result = AlignmentResult(
      pairs=pairs,
      shadow_only=[],
      production_only=[],
      mean_time_offset_ms=0.0,
      alignment_quality=0.95,
      method="gps",
    )

    validation = validate_alignment(result)

    assert validation["valid"] is True
    assert validation["pair_count"] == 10
    assert validation["coverage"] == 1.0

  def test_validate_poor_alignment(self):
    """Test validation of low-quality alignment."""
    from openpilot.tools.shadow.align import AlignedPair

    pairs = [
      AlignedPair(
        make_frame(i, float(i)),
        make_frame(i, float(i)),
        float(i * 100),  # Varying offsets
        0.3,  # Low quality
      )
      for i in range(5)
    ]

    result = AlignmentResult(
      pairs=pairs,
      shadow_only=[make_frame(i, float(i)) for i in range(10, 20)],
      production_only=[],
      mean_time_offset_ms=200.0,
      alignment_quality=0.3,
      method="timestamp",
    )

    validation = validate_alignment(result)

    assert validation["valid"] is False

  def test_validate_empty_alignment(self):
    """Test validation of empty alignment."""
    result = AlignmentResult(
      pairs=[],
      shadow_only=[],
      production_only=[],
      mean_time_offset_ms=0.0,
      alignment_quality=0.0,
      method="gps",
    )

    validation = validate_alignment(result)

    assert validation["valid"] is False
    assert validation["reason"] == "No pairs aligned"
