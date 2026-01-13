import os
from typing import cast

from openpilot.system.hardware.base import HardwareBase
from openpilot.system.hardware.tici.hardware import Tici
from openpilot.system.hardware.pc.hardware import Pc

TICI = os.path.isfile('/TICI')
AGNOS = os.path.isfile('/AGNOS')
PC = not TICI

# Check for NVIDIA GPU (DGX Spark, RTX, etc.)
NVIDIA_GPU = False
if PC:
  try:
    from openpilot.system.hardware.nvidia.gpu import is_nvidia_available

    NVIDIA_GPU = is_nvidia_available()
  except ImportError:
    pass

# Check for DGX Spark specifically
DGX_SPARK = False
if NVIDIA_GPU:
  try:
    from openpilot.system.hardware.nvidia.gpu import is_dgx_spark

    DGX_SPARK = is_dgx_spark()
  except ImportError:
    pass


if TICI:
  HARDWARE = cast(HardwareBase, Tici())
elif NVIDIA_GPU:
  from openpilot.system.hardware.nvidia.hardware import NvidiaPC

  HARDWARE = cast(HardwareBase, NvidiaPC())
else:
  HARDWARE = cast(HardwareBase, Pc())

# Shadow mode detection (for parallel testing devices)
from openpilot.system.hardware.shadow_mode import (
  SHADOW_MODE as SHADOW_MODE,
  is_shadow_mode as is_shadow_mode,
  panda_connected as panda_connected,
)
