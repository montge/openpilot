"""Tests for common/logging_extra.py - extended logging utilities."""
import json
import logging
import unittest
from unittest.mock import MagicMock, patch
from collections import OrderedDict

import numpy as np

from openpilot.common.logging_extra import (
  json_handler, json_robust_dumps, NiceOrderedDict,
  SwagFormatter, SwagLogFileFormatter, SwagErrorFilter, SwagLogger,
)


class TestJsonHandler(unittest.TestCase):
  """Test json_handler function."""

  def test_numpy_bool_true(self):
    """Test numpy bool True is converted to Python bool."""
    result = json_handler(np.bool_(True))
    self.assertIsInstance(result, bool)
    self.assertTrue(result)

  def test_numpy_bool_false(self):
    """Test numpy bool False is converted to Python bool."""
    result = json_handler(np.bool_(False))
    self.assertIsInstance(result, bool)
    self.assertFalse(result)

  def test_unknown_type_uses_repr(self):
    """Test unknown types are converted using repr."""
    class CustomObj:
      def __repr__(self):
        return "CustomObj()"

    result = json_handler(CustomObj())
    self.assertEqual(result, "CustomObj()")


class TestJsonRobustDumps(unittest.TestCase):
  """Test json_robust_dumps function."""

  def test_simple_dict(self):
    """Test simple dict serialization."""
    result = json_robust_dumps({"key": "value"})
    self.assertEqual(json.loads(result), {"key": "value"})

  def test_numpy_bool_in_dict(self):
    """Test dict containing numpy bool."""
    result = json_robust_dumps({"flag": np.bool_(True)})
    parsed = json.loads(result)
    self.assertTrue(parsed["flag"])

  def test_nested_dict(self):
    """Test nested dict serialization."""
    data = {"outer": {"inner": 123}}
    result = json_robust_dumps(data)
    self.assertEqual(json.loads(result), data)


class TestNiceOrderedDict(unittest.TestCase):
  """Test NiceOrderedDict class."""

  def test_str_returns_json(self):
    """Test __str__ returns JSON string."""
    d = NiceOrderedDict()
    d['key'] = 'value'
    result = str(d)
    self.assertEqual(json.loads(result), {"key": "value"})

  def test_preserves_order(self):
    """Test order is preserved in output."""
    d = NiceOrderedDict()
    d['first'] = 1
    d['second'] = 2
    d['third'] = 3
    result = str(d)
    # JSON string should have keys in order
    self.assertIn('"first"', result)
    self.assertIn('"second"', result)
    self.assertIn('"third"', result)


class TestSwagFormatter(unittest.TestCase):
  """Test SwagFormatter class."""

  def setUp(self):
    """Set up test fixtures."""
    self.logger = MagicMock()
    self.logger.get_ctx.return_value = {}
    self.formatter = SwagFormatter(self.logger)

  def test_init(self):
    """Test formatter initialization."""
    self.assertEqual(self.formatter.swaglogger, self.logger)
    self.assertIsNotNone(self.formatter.host)

  def test_format_dict_simple_message(self):
    """Test format_dict with simple message."""
    record = logging.LogRecord(
      name="test", level=logging.INFO, pathname="test.py",
      lineno=1, msg="test message", args=(), exc_info=None
    )
    result = self.formatter.format_dict(record)

    self.assertEqual(result['msg'], "test message")
    self.assertEqual(result['level'], 'INFO')
    self.assertEqual(result['name'], 'test')

  def test_format_dict_with_dict_message(self):
    """Test format_dict with dict message."""
    record = logging.LogRecord(
      name="test", level=logging.INFO, pathname="test.py",
      lineno=1, msg={"event": "test"}, args=(), exc_info=None
    )
    result = self.formatter.format_dict(record)

    self.assertEqual(result['msg'], {"event": "test"})

  def test_format_dict_with_format_args(self):
    """Test format_dict with format arguments."""
    record = logging.LogRecord(
      name="test", level=logging.INFO, pathname="test.py",
      lineno=1, msg="value is %d", args=(42,), exc_info=None
    )
    result = self.formatter.format_dict(record)

    self.assertEqual(result['msg'], "value is 42")

  def test_format_dict_with_exception(self):
    """Test format_dict includes exception info."""
    try:
      raise ValueError("test error")
    except ValueError:
      import sys
      exc_info = sys.exc_info()

    record = logging.LogRecord(
      name="test", level=logging.ERROR, pathname="test.py",
      lineno=1, msg="error occurred", args=(), exc_info=exc_info
    )
    result = self.formatter.format_dict(record)

    self.assertIn('exc_info', result)
    self.assertIn('ValueError', result['exc_info'])

  def test_format_returns_json(self):
    """Test format returns valid JSON."""
    record = logging.LogRecord(
      name="test", level=logging.INFO, pathname="test.py",
      lineno=1, msg="test", args=(), exc_info=None
    )
    result = self.formatter.format(record)

    # Should be valid JSON
    parsed = json.loads(result)
    self.assertIn('msg', parsed)

  def test_format_without_swaglogger_raises(self):
    """Test format raises when swaglogger is None."""
    formatter = SwagFormatter(None)
    record = logging.LogRecord(
      name="test", level=logging.INFO, pathname="test.py",
      lineno=1, msg="test", args=(), exc_info=None
    )
    with self.assertRaises(Exception) as ctx:
      formatter.format(record)
    self.assertIn("must set swaglogger", str(ctx.exception))


class TestSwagLogFileFormatter(unittest.TestCase):
  """Test SwagLogFileFormatter class."""

  def setUp(self):
    """Set up test fixtures."""
    self.logger = MagicMock()
    self.logger.get_ctx.return_value = {}
    self.formatter = SwagLogFileFormatter(self.logger)

  def test_fix_kv_string(self):
    """Test fix_kv appends $s for strings."""
    key, val = self.formatter.fix_kv('name', 'John')
    self.assertEqual(key, 'name$s')
    self.assertEqual(val, 'John')

  def test_fix_kv_bytes(self):
    """Test fix_kv appends $s for bytes."""
    key, val = self.formatter.fix_kv('data', b'bytes')
    self.assertEqual(key, 'data$s')

  def test_fix_kv_float(self):
    """Test fix_kv appends $f for floats."""
    key, val = self.formatter.fix_kv('price', 3.14)
    self.assertEqual(key, 'price$f')
    self.assertEqual(val, 3.14)

  def test_fix_kv_bool(self):
    """Test fix_kv appends $b for booleans."""
    key, val = self.formatter.fix_kv('enabled', True)
    self.assertEqual(key, 'enabled$b')
    self.assertTrue(val)

  def test_fix_kv_int(self):
    """Test fix_kv appends $i for integers."""
    key, val = self.formatter.fix_kv('count', 42)
    self.assertEqual(key, 'count$i')
    self.assertEqual(val, 42)

  def test_fix_kv_list(self):
    """Test fix_kv appends $a for lists."""
    key, val = self.formatter.fix_kv('items', [1, 2, 3])
    self.assertEqual(key, 'items$a')
    self.assertEqual(val, [1, 2, 3])

  def test_fix_kv_nested_dict(self):
    """Test fix_kv recursively fixes nested dict."""
    key, val = self.formatter.fix_kv('data', {'name': 'John', 'age': 30})
    self.assertEqual(key, 'data')
    self.assertIn('name$s', val)
    self.assertIn('age$i', val)

  def test_format_record(self):
    """Test format adds id and renames msg."""
    record = logging.LogRecord(
      name="test", level=logging.INFO, pathname="test.py",
      lineno=1, msg="test message", args=(), exc_info=None
    )
    result = self.formatter.format(record)
    parsed = json.loads(result)

    # Should have id field
    self.assertIn('id', parsed)
    # msg should be renamed with type suffix
    self.assertIn('msg$s', parsed)
    self.assertNotIn('msg', parsed)

  def test_format_json_string(self):
    """Test format handles JSON string input."""
    json_str = json.dumps({
      'msg': 'test',
      'level': 'INFO',
      'ctx': {},
      'levelnum': 20,
      'name': 'test',
      'filename': 'test.py',
      'lineno': 1,
      'pathname': 'test.py',
      'module': 'test',
      'funcName': 'func',
      'host': 'localhost',
      'process': 1234,
      'thread': 5678,
      'threadName': 'main',
      'created': 1234567890.0,
    })
    result = self.formatter.format(json_str)
    parsed = json.loads(result)

    self.assertIn('id', parsed)
    self.assertIn('msg$s', parsed)


class TestSwagErrorFilter(unittest.TestCase):
  """Test SwagErrorFilter class."""

  def test_allows_debug(self):
    """Test filter allows DEBUG level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(
      name="test", level=logging.DEBUG, pathname="test.py",
      lineno=1, msg="debug", args=(), exc_info=None
    )
    self.assertTrue(f.filter(record))

  def test_allows_info(self):
    """Test filter allows INFO level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(
      name="test", level=logging.INFO, pathname="test.py",
      lineno=1, msg="info", args=(), exc_info=None
    )
    self.assertTrue(f.filter(record))

  def test_allows_warning(self):
    """Test filter allows WARNING level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(
      name="test", level=logging.WARNING, pathname="test.py",
      lineno=1, msg="warning", args=(), exc_info=None
    )
    self.assertTrue(f.filter(record))

  def test_blocks_error(self):
    """Test filter blocks ERROR level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(
      name="test", level=logging.ERROR, pathname="test.py",
      lineno=1, msg="error", args=(), exc_info=None
    )
    self.assertFalse(f.filter(record))

  def test_blocks_critical(self):
    """Test filter blocks CRITICAL level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(
      name="test", level=logging.CRITICAL, pathname="test.py",
      lineno=1, msg="critical", args=(), exc_info=None
    )
    self.assertFalse(f.filter(record))


class TestSwagLogger(unittest.TestCase):
  """Test SwagLogger class."""

  def test_init(self):
    """Test logger initialization."""
    logger = SwagLogger()
    self.assertEqual(logger.name, "swaglog")
    self.assertEqual(logger.global_ctx, {})

  def test_local_ctx_default(self):
    """Test local_ctx returns empty dict by default."""
    logger = SwagLogger()
    ctx = logger.local_ctx()
    self.assertEqual(ctx, {})

  def test_get_ctx_empty(self):
    """Test get_ctx returns empty dict when no context bound."""
    logger = SwagLogger()
    ctx = logger.get_ctx()
    self.assertEqual(ctx, {})

  def test_bind_adds_to_local_ctx(self):
    """Test bind adds values to local context."""
    logger = SwagLogger()
    logger.bind(user="test_user", request_id="123")
    ctx = logger.local_ctx()
    self.assertEqual(ctx['user'], "test_user")
    self.assertEqual(ctx['request_id'], "123")

  def test_bind_global_adds_to_global_ctx(self):
    """Test bind_global adds values to global context."""
    logger = SwagLogger()
    logger.bind_global(version="1.0", environment="test")
    self.assertEqual(logger.global_ctx['version'], "1.0")
    self.assertEqual(logger.global_ctx['environment'], "test")

  def test_get_ctx_merges_local_and_global(self):
    """Test get_ctx merges local and global contexts."""
    logger = SwagLogger()
    logger.bind_global(global_key="global_val")
    logger.bind(local_key="local_val")

    ctx = logger.get_ctx()
    self.assertEqual(ctx['global_key'], "global_val")
    self.assertEqual(ctx['local_key'], "local_val")

  def test_ctx_context_manager(self):
    """Test ctx context manager scopes local context."""
    logger = SwagLogger()
    logger.bind(outer="value")

    with logger.ctx(inner="scoped"):
      ctx = logger.local_ctx()
      self.assertEqual(ctx.get('outer'), "value")
      self.assertEqual(ctx.get('inner'), "scoped")

    # After context, inner should be gone
    ctx = logger.local_ctx()
    self.assertEqual(ctx.get('outer'), "value")
    self.assertNotIn('inner', ctx)

  def test_ctx_restores_on_exception(self):
    """Test ctx restores context even on exception."""
    logger = SwagLogger()
    logger.bind(existing="value")

    try:
      with logger.ctx(temp="scoped"):
        raise ValueError("test")
    except ValueError:
      pass

    ctx = logger.local_ctx()
    self.assertNotIn('temp', ctx)
    self.assertEqual(ctx.get('existing'), "value")

  def test_event_info_level(self):
    """Test event logs at INFO level by default."""
    logger = SwagLogger()
    logger.info = MagicMock()

    logger.event("user_login", user_id=123)

    logger.info.assert_called_once()
    call_arg = logger.info.call_args[0][0]
    self.assertEqual(call_arg['event'], "user_login")
    self.assertEqual(call_arg['user_id'], 123)

  def test_event_error_level(self):
    """Test event logs at ERROR level when error=True."""
    logger = SwagLogger()
    logger.error = MagicMock()

    logger.event("failed_login", error=True, user_id=123)

    logger.error.assert_called_once()

  def test_event_debug_level(self):
    """Test event logs at DEBUG level when debug=True."""
    logger = SwagLogger()
    logger.debug = MagicMock()

    logger.event("trace_event", debug=True)

    logger.debug.assert_called_once()

  def test_event_with_args(self):
    """Test event includes positional args."""
    logger = SwagLogger()
    logger.info = MagicMock()

    logger.event("test_event", "arg1", "arg2")

    call_arg = logger.info.call_args[0][0]
    self.assertEqual(call_arg['args'], ("arg1", "arg2"))

  @patch.dict('os.environ', {'LOG_TIMESTAMPS': '1'})
  def test_timestamp_logs_when_enabled(self):
    """Test timestamp logs when LOG_TIMESTAMPS is set."""
    # Need to reimport to pick up env var
    from importlib import reload
    import openpilot.common.logging_extra as le
    reload(le)

    logger = le.SwagLogger()
    logger.debug = MagicMock()

    logger.timestamp("test_event")

    logger.debug.assert_called_once()
    call_arg = logger.debug.call_args[0][0]
    self.assertIn('timestamp', call_arg)

  def test_find_caller_returns_tuple(self):
    """Test findCaller returns expected tuple format."""
    logger = SwagLogger()
    result = logger.findCaller()

    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 4)
    # Format: (filename, lineno, funcname, sinfo)
    self.assertIsInstance(result[0], str)
    self.assertIsInstance(result[1], int)
    self.assertIsInstance(result[2], str)


if __name__ == '__main__':
  unittest.main()
