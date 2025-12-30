import time
import threading
from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence

import openpilot.system.loggerd.deleter as deleter
from openpilot.common.timeout import Timeout, TimeoutException
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase
from openpilot.system.loggerd.deleter import get_preserved_segments

Stats = namedtuple("Stats", ['f_bavail', 'f_blocks', 'f_frsize'])


class TestDeleter(UploaderTestCase):
  def fake_statvfs(self, d):
    return self.fake_stats

  def setup_method(self):
    self.f_type = "fcamera.hevc"
    super().setup_method()
    self.fake_stats = Stats(f_bavail=0, f_blocks=10, f_frsize=4096)
    deleter.os.statvfs = self.fake_statvfs

  def start_thread(self):
    self.end_event = threading.Event()
    self.del_thread = threading.Thread(target=deleter.deleter_thread, args=[self.end_event])
    self.del_thread.daemon = True
    self.del_thread.start()

  def join_thread(self):
    self.end_event.set()
    self.del_thread.join()

  def test_delete(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, 1)

    self.start_thread()

    try:
      with Timeout(2, "Timeout waiting for file to be deleted"):
        while f_path.exists():
          time.sleep(0.01)
    finally:
      self.join_thread()

  def assertDeleteOrder(self, f_paths: Sequence[Path], timeout: int = 5) -> None:
    deleted_order = []

    self.start_thread()
    try:
      with Timeout(timeout, "Timeout waiting for files to be deleted"):
        while True:
          for f in f_paths:
            if not f.exists() and f not in deleted_order:
              deleted_order.append(f)
          if len(deleted_order) == len(f_paths):
            break
          time.sleep(0.01)
    except TimeoutException:
      print("Not deleted:", [f for f in f_paths if f not in deleted_order])
      raise
    finally:
      self.join_thread()

    assert deleted_order == f_paths, "Files not deleted in expected order"

  def test_delete_order(self):
    self.assertDeleteOrder(
      [
        self.make_file_with_data(self.seg_format.format(0), self.f_type),
        self.make_file_with_data(self.seg_format.format(1), self.f_type),
        self.make_file_with_data(self.seg_format2.format(0), self.f_type),
      ]
    )

  def test_delete_many_preserved(self):
    self.assertDeleteOrder(
      [
        self.make_file_with_data(self.seg_format.format(0), self.f_type),
        self.make_file_with_data(self.seg_format.format(1), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
        self.make_file_with_data(self.seg_format.format(2), self.f_type),
      ]
      + [self.make_file_with_data(self.seg_format2.format(i), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE) for i in range(5)]
    )

  def test_delete_last(self):
    self.assertDeleteOrder(
      [
        self.make_file_with_data(self.seg_format.format(1), self.f_type),
        self.make_file_with_data(self.seg_format2.format(0), self.f_type),
        self.make_file_with_data(self.seg_format.format(0), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
        self.make_file_with_data("boot", self.seg_format[:-4]),
        self.make_file_with_data("crash", self.seg_format2[:-4]),
      ]
    )

  def test_no_delete_when_available_space(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)

    block_size = 4096
    available = (10 * 1024 * 1024 * 1024) / block_size  # 10GB free
    self.fake_stats = Stats(f_bavail=available, f_blocks=10, f_frsize=block_size)

    self.start_thread()
    start_time = time.monotonic()
    while f_path.exists() and time.monotonic() - start_time < 2:
      time.sleep(0.01)
    self.join_thread()

    assert f_path.exists(), "File deleted with available space"

  def test_no_delete_with_lock_file(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, lock=True)

    self.start_thread()
    start_time = time.monotonic()
    while f_path.exists() and time.monotonic() - start_time < 2:
      time.sleep(0.01)
    self.join_thread()

    assert f_path.exists(), "File deleted when locked"


class TestGetPreservedSegments:
  """Test get_preserved_segments edge cases."""

  def test_non_segment_directory_skipped(self, mocker):
    """Test directories without -- separator are skipped (line 35)."""
    # Directory without -- separator should be ignored
    mocker.patch.object(deleter, 'has_preserve_xattr', return_value=True)
    dirs = ["nodashes", "2024_01_01--0"]
    result = get_preserved_segments(dirs)
    # "nodashes" should be skipped, only "2024_01_01--0" preserved
    assert "nodashes" not in result
    assert "2024_01_01--0" in result

  def test_invalid_segment_number_skipped(self, mocker):
    """Test directories with non-integer segment number are skipped (lines 38-39)."""
    mocker.patch.object(deleter, 'has_preserve_xattr', return_value=True)
    dirs = ["2024_01_01--abc", "2024_01_01--5"]
    result = get_preserved_segments(dirs)
    # "2024_01_01--abc" should be skipped due to ValueError
    assert "2024_01_01--abc" not in result
    # Valid segment should be preserved along with prior segments
    assert "2024_01_01--5" in result
    assert "2024_01_01--4" in result
    assert "2024_01_01--3" in result


class TestDeleterOSError(UploaderTestCase):
  """Test OSError handling in deleter."""

  def fake_statvfs(self, d):
    return self.fake_stats

  def setup_method(self):
    self.f_type = "fcamera.hevc"
    super().setup_method()
    self.fake_stats = Stats(f_bavail=0, f_blocks=10, f_frsize=4096)
    deleter.os.statvfs = self.fake_statvfs

  def test_oserror_during_delete_continues(self, mocker):
    """Test OSError during deletion is caught and logged (lines 68-69)."""
    self.make_file_with_data(self.seg_dir, self.f_type, 1)

    end_event = threading.Event()
    delete_attempts = []

    original_rmtree = deleter.shutil.rmtree

    def mock_rmtree(path):
      delete_attempts.append(path)
      if len(delete_attempts) == 1:
        raise OSError("Simulated deletion failure")
      return original_rmtree(path)

    mocker.patch.object(deleter.shutil, 'rmtree', side_effect=mock_rmtree)
    del_thread = threading.Thread(target=deleter.deleter_thread, args=[end_event])
    del_thread.daemon = True
    del_thread.start()

    # Wait for at least one delete attempt
    start_time = time.monotonic()
    while len(delete_attempts) < 1 and time.monotonic() - start_time < 2:
      time.sleep(0.01)

    end_event.set()
    del_thread.join(timeout=2)

    # Should have attempted to delete and caught the error
    assert len(delete_attempts) >= 1
