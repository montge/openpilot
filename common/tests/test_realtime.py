import time
from unittest.mock import patch, MagicMock

import pytest

from openpilot.common.realtime import (
  DT_CTRL,
  DT_MDL,
  DT_HW,
  DT_DMON,
  Priority,
  set_core_affinity,
  config_realtime_process,
  Ratekeeper,
)


class TestConstants:
  def test_dt_ctrl(self):
    """Test that control loop time step is 10ms (100Hz)."""
    assert DT_CTRL == 0.01

  def test_dt_mdl(self):
    """Test that model time step is 50ms (20Hz)."""
    assert DT_MDL == 0.05

  def test_dt_hw(self):
    """Test that hardware time step is 500ms (2Hz)."""
    assert DT_HW == 0.5

  def test_dt_dmon(self):
    """Test that driver monitoring time step is 50ms (20Hz)."""
    assert DT_DMON == 0.05


class TestPriority:
  def test_ctrl_low(self):
    """Test CTRL_LOW priority value."""
    assert Priority.CTRL_LOW == 51

  def test_ctrl_high(self):
    """Test CTRL_HIGH priority value."""
    assert Priority.CTRL_HIGH == 53


class TestSetCoreAffinity:
  @patch('openpilot.common.realtime.PC', False)
  @patch('openpilot.common.realtime.sys.platform', 'linux')
  @patch('openpilot.common.realtime.os.sched_setaffinity')
  def test_set_core_affinity_linux_not_pc(self, mock_setaffinity):
    """Test that core affinity is set on Linux non-PC."""
    set_core_affinity([0, 1])
    mock_setaffinity.assert_called_once_with(0, [0, 1])

  @patch('openpilot.common.realtime.PC', True)
  @patch('openpilot.common.realtime.sys.platform', 'linux')
  @patch('openpilot.common.realtime.os.sched_setaffinity')
  def test_set_core_affinity_linux_pc(self, mock_setaffinity):
    """Test that core affinity is not set on Linux PC."""
    set_core_affinity([0, 1])
    mock_setaffinity.assert_not_called()

  @patch('openpilot.common.realtime.sys.platform', 'darwin')
  @patch('openpilot.common.realtime.os.sched_setaffinity')
  def test_set_core_affinity_non_linux(self, mock_setaffinity):
    """Test that core affinity is not set on non-Linux platforms."""
    set_core_affinity([0, 1])
    mock_setaffinity.assert_not_called()


class TestConfigRealtimeProcess:
  @patch('openpilot.common.realtime.set_core_affinity')
  @patch('openpilot.common.realtime.gc.disable')
  @patch('openpilot.common.realtime.PC', True)
  def test_config_realtime_process_pc(self, mock_gc_disable, mock_set_affinity):
    """Test config_realtime_process on PC (no scheduler change)."""
    config_realtime_process(cores=2, priority=53)

    mock_gc_disable.assert_called_once()
    mock_set_affinity.assert_called_once_with([2])

  @patch('openpilot.common.realtime.set_core_affinity')
  @patch('openpilot.common.realtime.gc.disable')
  @patch('openpilot.common.realtime.PC', True)
  def test_config_realtime_process_with_core_list(self, mock_gc_disable, mock_set_affinity):
    """Test config_realtime_process with list of cores."""
    config_realtime_process(cores=[0, 2, 4], priority=53)

    mock_set_affinity.assert_called_once_with([0, 2, 4])

  @patch('openpilot.common.realtime.set_core_affinity')
  @patch('openpilot.common.realtime.gc.disable')
  @patch('openpilot.common.realtime.os.sched_setscheduler')
  @patch('openpilot.common.realtime.PC', False)
  @patch('openpilot.common.realtime.sys.platform', 'linux')
  def test_config_realtime_process_linux_not_pc(self, mock_scheduler, mock_gc_disable, mock_set_affinity):
    """Test config_realtime_process on Linux non-PC."""
    import os
    config_realtime_process(cores=2, priority=53)

    mock_gc_disable.assert_called_once()
    mock_scheduler.assert_called_once()
    mock_set_affinity.assert_called_once_with([2])


class TestRatekeeper:
  def test_init(self):
    """Test Ratekeeper initialization."""
    rk = Ratekeeper(rate=100, print_delay_threshold=0.01)

    assert rk._interval == pytest.approx(0.01)
    assert rk._print_delay_threshold == 0.01
    assert rk._frame == 0
    assert rk._remaining == 0.0

  def test_init_default_threshold(self):
    """Test Ratekeeper initialization with default threshold."""
    rk = Ratekeeper(rate=50)
    assert rk._print_delay_threshold == 0.0

  def test_init_no_threshold(self):
    """Test Ratekeeper initialization with None threshold."""
    rk = Ratekeeper(rate=50, print_delay_threshold=None)
    assert rk._print_delay_threshold is None

  def test_frame_property(self):
    """Test frame property returns current frame count."""
    rk = Ratekeeper(rate=100)
    assert rk.frame == 0

    rk.monitor_time()
    assert rk.frame == 1

    rk.monitor_time()
    assert rk.frame == 2

  def test_remaining_property(self):
    """Test remaining property returns time until next frame."""
    rk = Ratekeeper(rate=100)
    # Initially remaining is 0
    assert rk.remaining == 0.0

    # After monitor_time, remaining should be set
    rk.monitor_time()
    # Remaining should be close to interval (minus small execution time)
    assert isinstance(rk.remaining, float)

  def test_lagging_property_not_lagging(self):
    """Test lagging property when not lagging."""
    rk = Ratekeeper(rate=100)

    # Add some values close to the interval
    for _ in range(10):
      rk.avg_dt.add_value(0.01)

    assert not rk.lagging

  def test_lagging_property_when_lagging(self):
    """Test lagging property when lagging."""
    rk = Ratekeeper(rate=100)

    # Add values much larger than the interval
    # Lagging when avg_dt > interval * (1 / 0.9) = 0.01 * 1.111 = 0.0111
    for _ in range(100):
      rk.avg_dt.add_value(0.02)

    assert rk.lagging

  def test_monitor_time_increments_frame(self):
    """Test that monitor_time increments frame counter."""
    rk = Ratekeeper(rate=100)

    for i in range(5):
      assert rk.frame == i
      rk.monitor_time()

    assert rk.frame == 5

  def test_monitor_time_first_call_initializes(self):
    """Test that first monitor_time call initializes timing."""
    rk = Ratekeeper(rate=100)

    assert rk._last_monitor_time == -1.
    assert rk._next_frame_time == -1.

    rk.monitor_time()

    assert rk._last_monitor_time > 0
    assert rk._next_frame_time > 0

  def test_monitor_time_updates_avg_dt(self):
    """Test that monitor_time updates the average dt."""
    rk = Ratekeeper(rate=100)

    rk.monitor_time()
    time.sleep(0.02)
    rk.monitor_time()

    # Average should include the sleep time
    avg = rk.avg_dt.get_average()
    assert avg > 0

  def test_monitor_time_returns_lagged_status(self):
    """Test that monitor_time returns whether the frame lagged."""
    rk = Ratekeeper(rate=1000, print_delay_threshold=0.001)

    # First call initializes
    lagged = rk.monitor_time()
    assert not lagged

    # Sleep longer than threshold to cause lag
    time.sleep(0.02)
    lagged = rk.monitor_time()
    assert lagged

  def test_monitor_time_no_print_when_threshold_none(self):
    """Test that monitor_time doesn't print when threshold is None."""
    rk = Ratekeeper(rate=100, print_delay_threshold=None)

    rk.monitor_time()
    time.sleep(0.1)  # Cause significant lag
    lagged = rk.monitor_time()

    # Should not print and should return False (no lag reporting)
    assert not lagged

  def test_keep_time_sleeps_when_remaining_positive(self):
    """Test that keep_time sleeps when ahead of schedule."""
    rk = Ratekeeper(rate=10)  # 100ms interval

    start = time.monotonic()
    rk.keep_time()
    elapsed = time.monotonic() - start

    # Should have slept close to the interval (minus small execution time)
    # Using a loose check since timing can vary
    assert elapsed >= 0.05  # At least half the interval

  def test_keep_time_returns_lagged_status(self):
    """Test that keep_time returns the lagged status from monitor_time."""
    rk = Ratekeeper(rate=1000, print_delay_threshold=0.001)

    # First call
    lagged = rk.keep_time()
    assert not lagged

  def test_keep_time_does_not_sleep_when_lagging(self):
    """Test that keep_time doesn't sleep much when already lagging."""
    rk = Ratekeeper(rate=100)

    # Initialize timing
    rk.monitor_time()

    # Simulate being behind schedule
    time.sleep(0.05)  # Sleep 5x the interval

    start = time.monotonic()
    rk.keep_time()
    elapsed = time.monotonic() - start

    # Should not sleep much since we're behind
    assert elapsed < 0.01

  def test_rate_calculation(self):
    """Test that the rate determines the correct interval."""
    rk_10hz = Ratekeeper(rate=10)
    assert rk_10hz._interval == pytest.approx(0.1)

    rk_100hz = Ratekeeper(rate=100)
    assert rk_100hz._interval == pytest.approx(0.01)

    rk_1000hz = Ratekeeper(rate=1000)
    assert rk_1000hz._interval == pytest.approx(0.001)

  def test_multiple_iterations(self):
    """Test Ratekeeper over multiple iterations."""
    rk = Ratekeeper(rate=100, print_delay_threshold=None)

    for i in range(10):
      assert rk.frame == i
      rk.keep_time()

    assert rk.frame == 10

  def test_remaining_updates_each_frame(self):
    """Test that remaining time is updated each frame."""
    rk = Ratekeeper(rate=100)

    rk.monitor_time()
    remaining1 = rk.remaining

    time.sleep(0.005)
    rk.monitor_time()
    remaining2 = rk.remaining

    # Both should be floats, values will differ based on timing
    assert isinstance(remaining1, float)
    assert isinstance(remaining2, float)
