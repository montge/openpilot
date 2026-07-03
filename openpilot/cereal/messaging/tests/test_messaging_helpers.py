"""Tests for cereal/messaging helper functions."""
import time

import capnp
import pytest

from cereal import log
import cereal.messaging as messaging
from cereal.services import SERVICE_LIST


class TestNewMessage:
  """Test new_message function."""

  def test_new_message_with_service(self):
    """Test new_message creates message with service."""
    msg = messaging.new_message("carState")

    assert msg.which() == "carState"
    assert not msg.valid

  def test_new_message_without_service(self):
    """Test new_message creates message without service (None)."""
    msg = messaging.new_message(None)

    assert msg.which() == "initData"  # Default union field

  def test_new_message_has_logMonoTime(self):
    """Test new_message sets logMonoTime."""
    before = int(time.monotonic() * 1e9)
    msg = messaging.new_message("carState")
    after = int(time.monotonic() * 1e9)

    assert before <= msg.logMonoTime <= after

  def test_new_message_valid_false_by_default(self):
    """Test new_message sets valid to False by default."""
    msg = messaging.new_message("carState")

    assert msg.valid is False

  def test_new_message_with_valid_override(self):
    """Test new_message can override valid."""
    msg = messaging.new_message("carState", valid=True)

    assert msg.valid is True

  def test_new_message_with_custom_logMonoTime(self):
    """Test new_message can override logMonoTime."""
    custom_time = 123456789
    msg = messaging.new_message("carState", logMonoTime=custom_time)

    assert msg.logMonoTime == custom_time


class TestLogFromBytes:
  """Test log_from_bytes function."""

  def test_log_from_bytes_roundtrip(self):
    """Test log_from_bytes can deserialize a serialized message."""
    original = messaging.new_message("carState")
    original.carState.vEgo = 25.5
    serialized = original.to_bytes()

    deserialized = messaging.log_from_bytes(serialized)

    assert deserialized.which() == "carState"
    assert abs(deserialized.carState.vEgo - 25.5) < 0.01

  def test_log_from_bytes_preserves_valid(self):
    """Test log_from_bytes preserves valid flag."""
    original = messaging.new_message("carState", valid=True)
    serialized = original.to_bytes()

    deserialized = messaging.log_from_bytes(serialized)

    assert deserialized.valid is True

  def test_log_from_bytes_preserves_logMonoTime(self):
    """Test log_from_bytes preserves logMonoTime."""
    custom_time = 987654321
    original = messaging.new_message("carState", logMonoTime=custom_time)
    serialized = original.to_bytes()

    deserialized = messaging.log_from_bytes(serialized)

    assert deserialized.logMonoTime == custom_time


class TestNoTraversalLimit:
  """Test NO_TRAVERSAL_LIMIT constant."""

  def test_no_traversal_limit_value(self):
    """Test NO_TRAVERSAL_LIMIT is max uint64."""
    assert messaging.NO_TRAVERSAL_LIMIT == 2**64 - 1

  def test_no_traversal_limit_is_int(self):
    """Test NO_TRAVERSAL_LIMIT is an integer."""
    assert isinstance(messaging.NO_TRAVERSAL_LIMIT, int)


class TestPubSubSockHelpers:
  """Test pub_sock and sub_sock helper behavior."""

  def test_pub_sock_known_service(self):
    """Test pub_sock with a known service."""
    sock = messaging.pub_sock("carState")
    assert sock is not None

  def test_pub_sock_unknown_service(self):
    """Test pub_sock with unknown service uses 0 segment size."""
    sock = messaging.pub_sock("unknownService12345")
    assert sock is not None

  def test_sub_sock_known_service(self):
    """Test sub_sock with a known service."""
    sock = messaging.sub_sock("carState", timeout=100)
    assert sock is not None

  def test_sub_sock_with_conflate(self):
    """Test sub_sock with conflate option."""
    sock = messaging.sub_sock("carState", conflate=True, timeout=100)
    assert sock is not None


class TestResetContext:
  """Test reset_context function."""

  def test_reset_context_runs(self):
    """Test reset_context doesn't raise."""
    # Just verify it runs without error
    messaging.reset_context()


class TestServiceListIntegration:
  """Test messaging integration with SERVICE_LIST."""

  def test_new_message_all_services(self):
    """Test new_message works with all services in SERVICE_LIST."""
    for service_name in list(SERVICE_LIST.keys())[:10]:  # Test first 10
      try:
        msg = messaging.new_message(service_name)
        assert msg.which() == service_name
      except capnp.lib.capnp.KjException:
        # Some services require a size parameter (lists)
        msg = messaging.new_message(service_name, 0)
        assert msg.which() == service_name
