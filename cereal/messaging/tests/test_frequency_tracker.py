"""Tests for FrequencyTracker class in cereal/messaging."""
import pytest

from cereal.messaging import FrequencyTracker


class TestFrequencyTrackerInit:
  """Test FrequencyTracker initialization."""

  def test_init_basic(self):
    """Test basic initialization."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    assert tracker.prev_time == 0.0
    assert tracker.min_freq > 0
    assert tracker.max_freq > 0

  def test_init_poll_mode(self):
    """Test initialization in poll mode."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=True)

    # In poll mode, min_freq and max_freq are based on same freq with 0.8/1.2 multipliers
    assert abs(tracker.min_freq - 20.0 * 0.8) < 0.01
    assert abs(tracker.max_freq - 20.0 * 1.2) < 0.01

  def test_init_non_poll_mode(self):
    """Test initialization in non-poll mode."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    # In non-poll mode with equal frequencies, min <= max
    assert tracker.min_freq <= tracker.max_freq

  def test_init_high_service_freq(self):
    """Test initialization with service_freq >= 2 * update_freq."""
    tracker = FrequencyTracker(service_freq=50.0, update_freq=20.0, is_poll=False)

    # min_freq should be update_freq * 0.8
    assert abs(tracker.min_freq - 20.0 * 0.8) < 0.01

  def test_init_high_update_freq(self):
    """Test initialization with update_freq >= 2 * service_freq."""
    tracker = FrequencyTracker(service_freq=10.0, update_freq=25.0, is_poll=False)

    assert tracker.min_freq > 0
    assert tracker.max_freq > 0

  def test_init_very_low_frequency(self):
    """Test initialization with very low frequency (clipped to 1)."""
    tracker = FrequencyTracker(service_freq=0.5, update_freq=0.5, is_poll=False)

    # freq is max(min(0.5, 0.5), 1) = 1.0, but max_freq = min(freq, 0.5) = 0.5
    # With 1.2 multiplier: 0.5 * 1.2 = 0.6
    assert tracker.max_freq > 0
    assert tracker.min_freq > 0


class TestFrequencyTrackerRecordRecvTime:
  """Test FrequencyTracker.record_recv_time method."""

  def test_first_record_updates_prev_time(self):
    """Test first record just sets prev_time."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    tracker.record_recv_time(1.0)

    assert tracker.prev_time == 1.0
    assert tracker.avg_dt.count == 0  # No dt recorded yet

  def test_second_record_adds_dt(self):
    """Test second record adds dt to averages."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    tracker.record_recv_time(1.0)
    tracker.record_recv_time(1.05)  # 50ms later

    assert tracker.prev_time == 1.05
    assert tracker.avg_dt.count == 1

  def test_multiple_records(self):
    """Test multiple records accumulate properly."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    # Start at time 1.0 to ensure prev_time > 1e-5 check passes
    for i in range(10):
      tracker.record_recv_time(1.0 + i * 0.05)

    assert tracker.avg_dt.count == 9  # First one doesn't add a dt


class TestFrequencyTrackerValid:
  """Test FrequencyTracker.valid property."""

  def test_valid_no_data(self):
    """Test valid is False when no data recorded."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    assert tracker.valid is False

  def test_valid_with_good_frequency(self):
    """Test valid is True with good frequency."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    # Simulate receiving at exactly 20Hz (50ms intervals)
    for i in range(50):
      tracker.record_recv_time(i * 0.05)

    assert tracker.valid is True

  def test_valid_with_bad_average_but_good_recent(self):
    """Test valid is True if recent frequency is good."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    # First receive at bad frequency (5Hz)
    for i in range(20):
      tracker.record_recv_time(i * 0.2)

    # Then receive at good frequency (20Hz)
    base_time = 4.0
    for i in range(30):
      tracker.record_recv_time(base_time + i * 0.05)

    # Recent frequency should be good even if average is mixed
    assert tracker.valid is True

  def test_invalid_with_slow_frequency(self):
    """Test valid is False with too slow frequency."""
    tracker = FrequencyTracker(service_freq=20.0, update_freq=20.0, is_poll=False)

    # Simulate receiving at 5Hz (200ms intervals) - way too slow
    for i in range(20):
      tracker.record_recv_time(i * 0.2)

    assert tracker.valid is False


class TestFrequencyTrackerEdgeCases:
  """Test FrequencyTracker edge cases."""

  def test_poll_mode_bounds(self):
    """Test poll mode sets min and max correctly."""
    tracker = FrequencyTracker(service_freq=100.0, update_freq=20.0, is_poll=True)

    # In poll mode, freq is clamped to [1, update_freq] = [1, 20]
    freq = min(max(100.0, 20.0), 1.0)  # This would be 100, but clamped
    expected_freq = max(min(100.0, 20.0), 1.)  # = 20
    assert tracker.min_freq == expected_freq * 0.8
    assert tracker.max_freq == expected_freq * 1.2

  def test_intermediate_frequencies(self):
    """Test non-poll mode with intermediate frequencies."""
    # Case where neither service_freq >= 2*update_freq nor update_freq >= 2*service_freq
    tracker = FrequencyTracker(service_freq=15.0, update_freq=20.0, is_poll=False)

    # freq = max(min(15, 20), 1) = 15
    # max_freq = min(15, 20) = 15, so max_freq = 15 * 1.2
    # min_freq = min(15, 15/2) = 7.5, so min_freq = 7.5 * 0.8
    assert abs(tracker.max_freq - 15.0 * 1.2) < 0.01
    assert abs(tracker.min_freq - 7.5 * 0.8) < 0.01
