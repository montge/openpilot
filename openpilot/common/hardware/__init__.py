import os
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.comma.hardware import HardwareComma
from openpilot.common.hardware.pc.hardware import HardwarePc

AGNOS = os.path.isfile('/AGNOS')
COMMA_HARDWARE = AGNOS
PC = not COMMA_HARDWARE

# Check for NVIDIA GPU (DGX Spark, RTX, etc.)
NVIDIA_GPU = False
if PC:
  try:
    from openpilot.common.hardware.nvidia.gpu import is_nvidia_available

    NVIDIA_GPU = is_nvidia_available()
  except ImportError:
    pass

# Check for DGX Spark specifically
DGX_SPARK = False
if NVIDIA_GPU:
  try:
    from openpilot.common.hardware.nvidia.gpu import is_dgx_spark

    DGX_SPARK = is_dgx_spark()
  except ImportError:
    pass


if COMMA_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareComma())
elif NVIDIA_GPU:
  from openpilot.common.hardware.nvidia.hardware import NvidiaPC

  HARDWARE = cast(HardwareBase, NvidiaPC())
else:
  HARDWARE = cast(HardwareBase, HardwarePc())

# Shadow mode detection (for parallel testing devices)
from openpilot.common.hardware.shadow_mode import (
  SHADOW_MODE as SHADOW_MODE,
  is_shadow_mode as is_shadow_mode,
  panda_connected as panda_connected,
)
