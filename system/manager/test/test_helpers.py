"""Tests for system/manager/helpers.py - manager helper functions."""

from openpilot.common.params import Params
from openpilot.system.manager.helpers import write_onroad_params


class TestWriteOnroadParams:
  """Test write_onroad_params function."""

  def test_write_onroad_params_started_true(self):
    """Test write_onroad_params with started=True."""
    params = Params()

    write_onroad_params(True, params)

    assert params.get_bool("IsOnroad") is True
    assert params.get_bool("IsOffroad") is False

  def test_write_onroad_params_started_false(self):
    """Test write_onroad_params with started=False."""
    params = Params()

    write_onroad_params(False, params)

    assert params.get_bool("IsOnroad") is False
    assert params.get_bool("IsOffroad") is True
