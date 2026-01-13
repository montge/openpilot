"""Shadow mode detection for parallel testing devices.

Shadow mode enables running openpilot on a secondary device (like OnePlus 6)
alongside a production device for comparison testing. In shadow mode:
- All sensors and cameras are active
- Model inference runs normally
- Control commands are computed but NOT sent to actuators
- All outputs are logged for offline comparison

Detection priority:
1. SHADOW_MODE=1 environment variable (highest priority)
2. OnePlus 6 device without panda connection
3. SHADOW_DEVICE=1 with no panda (generic override)

Usage:
  from openpilot.system.hardware.shadow_mode import is_shadow_mode, SHADOW_MODE

  if SHADOW_MODE:
    # Skip actuator publishing
    pass
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _check_panda_connected() -> bool:
  """Check if panda hardware is connected (USB or internal).

  Returns:
    True if any panda is detected, False otherwise.
  """
  try:
    from panda import Panda

    serials = Panda.list()
    return len(serials) > 0
  except ImportError:
    # panda library not available (e.g., in some test environments)
    return False
  except Exception:
    # Any other error (USB issues, etc.) - assume no panda
    return False


def panda_connected() -> bool:
  """Check if panda hardware is connected.

  This is a wrapper that can be mocked in tests.
  """
  return _check_panda_connected()


def clear_panda_cache() -> None:
  """Clear the panda connection cache.

  Call this after connecting/disconnecting panda to refresh detection.
  """
  _check_panda_connected.cache_clear()


@lru_cache(maxsize=1)
def _get_device_model() -> str:
  """Get the device model from device tree.

  Returns:
    Device model string or empty string if not available.
  """
  try:
    with open("/sys/firmware/devicetree/base/model") as f:
      return f.read().strip("\x00")
  except FileNotFoundError:
    return ""
  except Exception:
    return ""


def is_oneplus6() -> bool:
  """Check if running on OnePlus 6 hardware.

  The OnePlus 6 uses Snapdragon 845 (same as comma two) and is
  an ideal shadow device due to its camera and compute capabilities.
  """
  model = _get_device_model()
  # OnePlus 6 device tree contains "OnePlus" or specific model
  return "OnePlus" in model or "enchilada" in model.lower()


def is_shadow_device() -> bool:
  """Check if this hardware is designated as a shadow device.

  Shadow devices are secondary devices used for parallel testing.
  They run the full pipeline but never send actuator commands.
  """
  # Environment override
  if os.environ.get("SHADOW_DEVICE") == "1":
    return True

  # OnePlus 6 is a known shadow device platform
  if is_oneplus6():
    return True

  return False


@lru_cache(maxsize=1)
def _compute_shadow_mode() -> bool:
  """Compute whether shadow mode should be active.

  Shadow mode is active when:
  1. SHADOW_MODE=1 environment variable is set (explicit override)
  2. Running on a shadow device (OnePlus 6) without panda connection
  3. SHADOW_DEVICE=1 is set and no panda is connected

  Returns:
    True if shadow mode should be active.
  """
  # Priority 1: Explicit environment override
  shadow_env = os.environ.get("SHADOW_MODE", "").lower()
  if shadow_env == "1" or shadow_env == "true":
    return True
  if shadow_env == "0" or shadow_env == "false":
    return False

  # Priority 2: Shadow device without panda
  if is_shadow_device() and not panda_connected():
    return True

  # Priority 3: Generic shadow device override without panda
  if os.environ.get("SHADOW_DEVICE") == "1" and not panda_connected():
    return True

  return False


def is_shadow_mode() -> bool:
  """Check if running in shadow mode.

  Shadow mode disables all actuator outputs while keeping the full
  sensor and control pipeline active for comparison testing.

  This function is cached - call clear_shadow_mode_cache() to refresh.
  """
  return _compute_shadow_mode()


def clear_shadow_mode_cache() -> None:
  """Clear the shadow mode detection cache.

  Call this after environment changes or panda connection changes.
  """
  _compute_shadow_mode.cache_clear()
  _check_panda_connected.cache_clear()
  _get_device_model.cache_clear()


# Module-level constant for fast access (computed once at import)
# Use is_shadow_mode() function for dynamic checks
SHADOW_MODE = is_shadow_mode()
