# fork: upstream deleted this file when it replaced pytest with a unittest-based
# harness (openpilot/common/test.py OpenpilotTestCase). The fork keeps pytest as its
# runner, so the per-test isolation that used to live here is restored as fork-owned.
#
# Tests that inherit OpenpilotTestCase already set up their own prefix/env isolation
# inside TestCase.run(), so the autouse fixture below deliberately steps aside for
# them rather than nesting a second OpenpilotPrefix around the same test.
import contextlib
import gc
import os
import pytest

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.manager import manager
from openpilot.common.hardware import COMMA_HARDWARE, HARDWARE

# these are heavy CI-only tests, invoked explicitly in .github/workflows/tests.yaml
collect_ignore = [
  "openpilot/selfdrive/test/process_replay/test_processes.py",

  "openpilot/tools/sim/",
]


def pytest_sessionstart(session):
  # TODO: fix tests and enable test order randomization
  if session.config.pluginmanager.hasplugin('randomly'):
    session.config.option.randomly_reorganize = False


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_call(item):
  # ensure we run as a hook after capturemanager's
  if item.get_closest_marker("nocapture") is not None:
    capmanager = item.config.pluginmanager.getplugin("capturemanager")
    with capmanager.global_and_fixture_disabled():
      yield
  else:
    yield


@contextlib.contextmanager
def clean_env():
  starting_env = dict(os.environ)
  yield
  os.environ.clear()
  os.environ.update(starting_env)


def _manages_own_prefix(request) -> bool:
  cls = getattr(request.node, "cls", None)
  return cls is not None and isinstance(cls, type) and issubclass(cls, OpenpilotTestCase)


@pytest.fixture(scope="function", autouse=True)
def openpilot_function_fixture(request):
  if _manages_own_prefix(request):
    # OpenpilotTestCase.run() enters its own clean_env + OpenpilotPrefix
    yield
    return

  with clean_env():
    # setup a clean environment for each test
    with OpenpilotPrefix(shared_download_cache=request.node.get_closest_marker("shared_download_cache") is not None):
      prefix = os.environ["OPENPILOT_PREFIX"]

      yield

      # ensure the test doesn't change the prefix
      assert "OPENPILOT_PREFIX" in os.environ and prefix == os.environ["OPENPILOT_PREFIX"]

    # cleanup any started processes
    manager.manager_cleanup()

    # some processes disable gc for performance, re-enable here
    if not gc.isenabled():
      gc.enable()
      gc.collect()


# If you use setUpClass, the environment variables won't be cleared properly,
# so we need to hook both the function and class pytest fixtures
@pytest.fixture(scope="class", autouse=True)
def openpilot_class_fixture():
  with clean_env():
    yield


@pytest.fixture(scope="function")
def comma_hardware_setup_fixture(openpilot_function_fixture):
  """Ensure a consistent state for tests on-device. Needs the openpilot function fixture to run first."""
  HARDWARE.initialize_hardware()
  HARDWARE.set_power_save(False)
  os.system("pkill -9 -f athena")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
  # fork: upstream replaced the `tici` marker with OpenpilotTestCase.COMMA_HARDWARE_TEST,
  # which self-skips off-device. Plain pytest-style classes in the fork can still opt in
  # by setting COMMA_HARDWARE_TEST on the class.
  skipper = pytest.mark.skip(reason="Skipping comma hardware test on PC")
  for item in items:
    if getattr(getattr(item, "cls", None), "COMMA_HARDWARE_TEST", False):
      if not COMMA_HARDWARE:
        item.add_marker(skipper)
      else:
        item.fixturenames.append('comma_hardware_setup_fixture')

    if "xdist_group_class_property" in item.keywords:
      class_property_name = item.get_closest_marker('xdist_group_class_property').args[0]
      class_property_value = getattr(item.cls, class_property_name)
      item.add_marker(pytest.mark.xdist_group(class_property_value))
