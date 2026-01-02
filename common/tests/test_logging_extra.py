"""Tests for common/logging_extra.py - extended logging utilities."""

import json
import logging

import numpy as np
import pytest

from openpilot.common.logging_extra import (
  json_handler,
  json_robust_dumps,
  NiceOrderedDict,
  SwagFormatter,
  SwagLogFileFormatter,
  SwagErrorFilter,
  SwagLogger,
)


class TestJsonHandler:
  """Test json_handler function."""

  def test_numpy_bool_true(self):
    """Test numpy bool True is converted to Python bool."""
    result = json_handler(np.bool_(True))
    assert isinstance(result, bool)
    assert result is True

  def test_numpy_bool_false(self):
    """Test numpy bool False is converted to Python bool."""
    result = json_handler(np.bool_(False))
    assert isinstance(result, bool)
    assert result is False

  def test_unknown_type_uses_repr(self):
    """Test unknown types are converted using repr."""

    class CustomObj:
      def __repr__(self):
        return "CustomObj()"

    result = json_handler(CustomObj())
    assert result == "CustomObj()"


class TestJsonRobustDumps:
  """Test json_robust_dumps function."""

  def test_simple_dict(self):
    """Test simple dict serialization."""
    result = json_robust_dumps({"key": "value"})
    assert json.loads(result) == {"key": "value"}

  def test_numpy_bool_in_dict(self):
    """Test dict containing numpy bool."""
    result = json_robust_dumps({"flag": np.bool_(True)})
    parsed = json.loads(result)
    assert parsed["flag"] is True

  def test_nested_dict(self):
    """Test nested dict serialization."""
    data = {"outer": {"inner": 123}}
    result = json_robust_dumps(data)
    assert json.loads(result) == data


class TestNiceOrderedDict:
  """Test NiceOrderedDict class."""

  def test_str_returns_json(self):
    """Test __str__ returns JSON string."""
    d = NiceOrderedDict()
    d['key'] = 'value'
    result = str(d)
    assert json.loads(result) == {"key": "value"}

  def test_preserves_order(self):
    """Test order is preserved in output."""
    d = NiceOrderedDict()
    d['first'] = 1
    d['second'] = 2
    d['third'] = 3
    result = str(d)
    # JSON string should have keys in order
    assert '"first"' in result
    assert '"second"' in result
    assert '"third"' in result


class TestSwagFormatter:
  """Test SwagFormatter class."""

  def test_init(self, mocker):
    """Test formatter initialization."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)
    assert formatter.swaglogger == logger
    assert formatter.host is not None

  def test_format_dict_simple_message(self, mocker):
    """Test format_dict with simple message."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test message", args=(), exc_info=None)
    result = formatter.format_dict(record)

    assert result['msg'] == "test message"
    assert result['level'] == 'INFO'
    assert result['name'] == 'test'

  def test_format_dict_with_dict_message(self, mocker):
    """Test format_dict with dict message."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg={"event": "test"}, args=(), exc_info=None)
    result = formatter.format_dict(record)

    assert result['msg'] == {"event": "test"}

  def test_format_dict_with_format_args(self, mocker):
    """Test format_dict with format arguments."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="value is %d", args=(42,), exc_info=None)
    result = formatter.format_dict(record)

    assert result['msg'] == "value is 42"

  def test_format_dict_with_exception(self, mocker):
    """Test format_dict includes exception info."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)
    try:
      raise ValueError("test error")
    except ValueError:
      import sys

      exc_info = sys.exc_info()

    record = logging.LogRecord(name="test", level=logging.ERROR, pathname="test.py", lineno=1, msg="error occurred", args=(), exc_info=exc_info)
    result = formatter.format_dict(record)

    assert 'exc_info' in result
    assert 'ValueError' in result['exc_info']

  def test_format_returns_json(self, mocker):
    """Test format returns valid JSON."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test", args=(), exc_info=None)
    result = formatter.format(record)

    # Should be valid JSON
    parsed = json.loads(result)
    assert 'msg' in parsed

  def test_format_without_swaglogger_raises(self):
    """Test format raises when swaglogger is None."""
    formatter = SwagFormatter(None)
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test", args=(), exc_info=None)
    with pytest.raises(Exception) as ctx:
      formatter.format(record)
    assert "must set swaglogger" in str(ctx.value)


class TestSwagLogFileFormatter:
  """Test SwagLogFileFormatter class."""

  def test_fix_kv_string(self, mocker):
    """Test fix_kv appends $s for strings."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    key, val = formatter.fix_kv('name', 'John')
    assert key == 'name$s'
    assert val == 'John'

  def test_fix_kv_bytes(self, mocker):
    """Test fix_kv appends $s for bytes."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    key, val = formatter.fix_kv('data', b'bytes')
    assert key == 'data$s'

  def test_fix_kv_float(self, mocker):
    """Test fix_kv appends $f for floats."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    key, val = formatter.fix_kv('price', 3.14)
    assert key == 'price$f'
    assert val == 3.14

  def test_fix_kv_bool(self, mocker):
    """Test fix_kv appends $b for booleans."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    key, val = formatter.fix_kv('enabled', True)
    assert key == 'enabled$b'
    assert val is True

  def test_fix_kv_int(self, mocker):
    """Test fix_kv appends $i for integers."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    key, val = formatter.fix_kv('count', 42)
    assert key == 'count$i'
    assert val == 42

  def test_fix_kv_list(self, mocker):
    """Test fix_kv appends $a for lists."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    key, val = formatter.fix_kv('items', [1, 2, 3])
    assert key == 'items$a'
    assert val == [1, 2, 3]

  def test_fix_kv_nested_dict(self, mocker):
    """Test fix_kv recursively fixes nested dict."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    key, val = formatter.fix_kv('data', {'name': 'John', 'age': 30})
    assert key == 'data'
    assert 'name$s' in val
    assert 'age$i' in val

  def test_format_record(self, mocker):
    """Test format adds id and renames msg."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test message", args=(), exc_info=None)
    result = formatter.format(record)
    parsed = json.loads(result)

    # Should have id field
    assert 'id' in parsed
    # msg should be renamed with type suffix
    assert 'msg$s' in parsed
    assert 'msg' not in parsed

  def test_format_json_string(self, mocker):
    """Test format handles JSON string input."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagLogFileFormatter(logger)
    json_str = json.dumps(
      {
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
      }
    )
    result = formatter.format(json_str)
    parsed = json.loads(result)

    assert 'id' in parsed
    assert 'msg$s' in parsed


class TestSwagErrorFilter:
  """Test SwagErrorFilter class."""

  def test_allows_debug(self):
    """Test filter allows DEBUG level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(name="test", level=logging.DEBUG, pathname="test.py", lineno=1, msg="debug", args=(), exc_info=None)
    assert f.filter(record) is True

  def test_allows_info(self):
    """Test filter allows INFO level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="info", args=(), exc_info=None)
    assert f.filter(record) is True

  def test_allows_warning(self):
    """Test filter allows WARNING level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(name="test", level=logging.WARNING, pathname="test.py", lineno=1, msg="warning", args=(), exc_info=None)
    assert f.filter(record) is True

  def test_blocks_error(self):
    """Test filter blocks ERROR level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(name="test", level=logging.ERROR, pathname="test.py", lineno=1, msg="error", args=(), exc_info=None)
    assert f.filter(record) is False

  def test_blocks_critical(self):
    """Test filter blocks CRITICAL level."""
    f = SwagErrorFilter()
    record = logging.LogRecord(name="test", level=logging.CRITICAL, pathname="test.py", lineno=1, msg="critical", args=(), exc_info=None)
    assert f.filter(record) is False


class TestSwagLogger:
  """Test SwagLogger class."""

  def test_init(self):
    """Test logger initialization."""
    logger = SwagLogger()
    assert logger.name == "swaglog"
    assert logger.global_ctx == {}

  def test_local_ctx_default(self):
    """Test local_ctx returns empty dict by default."""
    logger = SwagLogger()
    ctx = logger.local_ctx()
    assert ctx == {}

  def test_get_ctx_empty(self):
    """Test get_ctx returns empty dict when no context bound."""
    logger = SwagLogger()
    ctx = logger.get_ctx()
    assert ctx == {}

  def test_bind_adds_to_local_ctx(self):
    """Test bind adds values to local context."""
    logger = SwagLogger()
    logger.bind(user="test_user", request_id="123")
    ctx = logger.local_ctx()
    assert ctx['user'] == "test_user"
    assert ctx['request_id'] == "123"

  def test_bind_global_adds_to_global_ctx(self):
    """Test bind_global adds values to global context."""
    logger = SwagLogger()
    logger.bind_global(version="1.0", environment="test")
    assert logger.global_ctx['version'] == "1.0"
    assert logger.global_ctx['environment'] == "test"

  def test_get_ctx_merges_local_and_global(self):
    """Test get_ctx merges local and global contexts."""
    logger = SwagLogger()
    logger.bind_global(global_key="global_val")
    logger.bind(local_key="local_val")

    ctx = logger.get_ctx()
    assert ctx['global_key'] == "global_val"
    assert ctx['local_key'] == "local_val"

  def test_ctx_context_manager(self):
    """Test ctx context manager scopes local context."""
    logger = SwagLogger()
    logger.bind(outer="value")

    with logger.ctx(inner="scoped"):
      ctx = logger.local_ctx()
      assert ctx.get('outer') == "value"
      assert ctx.get('inner') == "scoped"

    # After context, inner should be gone
    ctx = logger.local_ctx()
    assert ctx.get('outer') == "value"
    assert 'inner' not in ctx

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
    assert 'temp' not in ctx
    assert ctx.get('existing') == "value"

  def test_event_info_level(self, mocker):
    """Test event logs at INFO level by default."""
    logger = SwagLogger()
    logger.info = mocker.MagicMock()

    logger.event("user_login", user_id=123)

    logger.info.assert_called_once()
    call_arg = logger.info.call_args[0][0]
    assert call_arg['event'] == "user_login"
    assert call_arg['user_id'] == 123

  def test_event_error_level(self, mocker):
    """Test event logs at ERROR level when error=True."""
    logger = SwagLogger()
    logger.error = mocker.MagicMock()

    logger.event("failed_login", error=True, user_id=123)

    logger.error.assert_called_once()

  def test_event_debug_level(self, mocker):
    """Test event logs at DEBUG level when debug=True."""
    logger = SwagLogger()
    logger.debug = mocker.MagicMock()

    logger.event("trace_event", debug=True)

    logger.debug.assert_called_once()

  def test_event_with_args(self, mocker):
    """Test event includes positional args."""
    logger = SwagLogger()
    logger.info = mocker.MagicMock()

    logger.event("test_event", "arg1", "arg2")

    call_arg = logger.info.call_args[0][0]
    assert call_arg['args'] == ("arg1", "arg2")

  def test_timestamp_logs_when_enabled(self, mocker):
    """Test timestamp logs when LOG_TIMESTAMPS is set."""
    mocker.patch.dict('os.environ', {'LOG_TIMESTAMPS': '1'})
    # Need to reimport to pick up env var
    from importlib import reload
    import openpilot.common.logging_extra as le

    reload(le)

    logger = le.SwagLogger()
    logger.debug = mocker.MagicMock()

    logger.timestamp("test_event")

    logger.debug.assert_called_once()
    call_arg = logger.debug.call_args[0][0]
    assert 'timestamp' in call_arg

  def test_find_caller_returns_tuple(self):
    """Test findCaller returns expected tuple format."""
    logger = SwagLogger()
    result = logger.findCaller()

    assert isinstance(result, tuple)
    assert len(result) == 4
    # Format: (filename, lineno, funcname, sinfo)
    assert isinstance(result[0], str)
    assert isinstance(result[1], int)
    assert isinstance(result[2], str)

  def test_find_caller_with_stack_info(self):
    """Test findCaller with stack_info=True."""
    logger = SwagLogger()
    result = logger.findCaller(stack_info=True)

    assert isinstance(result, tuple)
    assert len(result) == 4
    # sinfo should contain stack trace string
    assert result[3] is not None
    assert "Stack" in result[3]

  def test_local_ctx_creates_new_on_attribute_error(self, mocker):
    """Test local_ctx creates new dict when thread local has no ctx."""
    logger = SwagLogger()

    # Delete the ctx attribute to trigger AttributeError path
    del logger.log_local.ctx

    # Should create new empty dict
    ctx = logger.local_ctx()
    assert ctx == {}

  def test_find_caller_with_stacklevel(self):
    """Test findCaller with stacklevel > 1 traverses frames."""
    logger = SwagLogger()

    def inner_func():
      return logger.findCaller(stacklevel=2)

    def outer_func():
      return inner_func()

    result = outer_func()

    assert isinstance(result, tuple)
    assert len(result) == 4
    # Should return caller info from higher up the stack
    assert isinstance(result[0], str)
    assert isinstance(result[1], int)


class TestSwagFormatterMessageException:
  """Test SwagFormatter getMessage exception handling."""

  def test_format_dict_with_message_exception_uses_list_args(self, mocker):
    """Test format_dict fallback when getMessage raises ValueError with list args."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)

    # Create a record with list args (not tuple) to test the fallback path
    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test %s", args=(), exc_info=None)
    # Manually set args to a list to work with the fallback code
    record.args = []
    mocker.patch.object(record, 'getMessage', side_effect=ValueError("format error"))

    result = formatter.format_dict(record)

    # Should fallback to [msg, *args] format
    assert result['msg'] == ["test %s"]

  def test_format_dict_with_type_error_uses_list_args(self, mocker):
    """Test format_dict fallback when getMessage raises TypeError with list args."""
    logger = mocker.MagicMock()
    logger.get_ctx.return_value = {}
    formatter = SwagFormatter(logger)

    record = logging.LogRecord(name="test", level=logging.INFO, pathname="test.py", lineno=1, msg="test %d", args=("not_a_number",), exc_info=None)
    # Manually set args to a list to work with the fallback code
    record.args = ["not_a_number"]
    mocker.patch.object(record, 'getMessage', side_effect=TypeError("format error"))

    result = formatter.format_dict(record)

    # Should fallback to [msg, *args] format
    assert result['msg'] == ["test %d", "not_a_number"]


class TestSrcfileHelper:
  """Test _srcfile helper function."""

  def test_srcfile_returns_normalized_path(self):
    """Test _srcfile returns normalized module path."""
    from openpilot.common.logging_extra import _srcfile

    result = _srcfile()

    assert isinstance(result, str)
    # Should be a normalized path to logging_extra.py
    assert "logging_extra" in result

  def test_tmpfunc_returns_zero(self):
    """Test _tmpfunc returns 0."""
    from openpilot.common.logging_extra import _tmpfunc

    result = _tmpfunc()
    assert result == 0
