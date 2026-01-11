"""
NVIDIA PC hardware abstraction.

Extends the base PC hardware class with NVIDIA GPU capabilities
for development and experimentation with GPU acceleration.
"""

from cereal import log
from openpilot.system.hardware.base import HardwareBase, ThermalConfig, ThermalZone
from openpilot.system.hardware.nvidia.gpu import (
  is_nvidia_available,
  is_dgx_spark,
  get_nvidia_gpus,
  get_best_gpu,
  GPUInfo,
)

NetworkType = log.DeviceState.NetworkType


class NvidiaPC(HardwareBase):
  """
  Hardware abstraction for PCs with NVIDIA GPUs.

  Provides GPU detection, thermal monitoring, and capability
  queries for NVIDIA GPUs including DGX Spark.
  """

  def __init__(self):
    self._gpu_info: GPUInfo | None = None
    self._refresh_gpu_info()

  def _refresh_gpu_info(self) -> None:
    """Refresh cached GPU information."""
    self._gpu_info = get_best_gpu()

  def get_device_type(self) -> str:
    """Get device type string."""
    if is_dgx_spark():
      return "dgx_spark"
    elif is_nvidia_available():
      return "nvidia_pc"
    else:
      return "pc"

  def get_network_type(self):
    return NetworkType.wifi

  def has_nvidia_gpu(self) -> bool:
    """Check if NVIDIA GPU is available."""
    return is_nvidia_available()

  def is_dgx_spark(self) -> bool:
    """Check if running on DGX Spark."""
    return is_dgx_spark()

  def get_gpu_info(self) -> GPUInfo | None:
    """Get information about the best available GPU."""
    return self._gpu_info

  def get_all_gpus(self) -> list[GPUInfo]:
    """Get information about all available GPUs."""
    return get_nvidia_gpus()

  def get_gpu_usage_percent(self) -> int:
    """Get GPU utilization percentage."""
    if self._gpu_info is None:
      return 0

    # Try to query GPU utilization via nvidia-smi
    try:
      from openpilot.system.hardware.nvidia.gpu import _run_nvidia_smi

      output = _run_nvidia_smi(['--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', f'--id={self._gpu_info.index}'])
      if output:
        return int(output.strip())
    except (ValueError, TypeError):
      pass
    return 0

  def get_gpu_memory_usage_percent(self) -> int:
    """Get GPU memory utilization percentage."""
    if self._gpu_info is None:
      return 0

    try:
      from openpilot.system.hardware.nvidia.gpu import _run_nvidia_smi

      output = _run_nvidia_smi(['--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits', f'--id={self._gpu_info.index}'])
      if output:
        used, total = map(int, output.strip().split(','))
        return int(100 * used / total) if total > 0 else 0
    except (ValueError, TypeError):
      pass
    return 0

  def get_thermal_config(self) -> ThermalConfig:
    """Get thermal configuration including GPU temps."""
    config = ThermalConfig()

    # Try to add GPU thermal zone
    if is_nvidia_available():
      # NVIDIA GPUs report temp via nvidia-smi, not thermal_zone
      # We'll create a virtual thermal zone that queries nvidia-smi
      config.gpu = [NvidiaGPUThermalZone(self._gpu_info.index if self._gpu_info else 0)]

    return config

  def get_gpu_temperature(self) -> float:
    """Get GPU temperature in Celsius."""
    if self._gpu_info is None:
      return 0.0

    try:
      from openpilot.system.hardware.nvidia.gpu import _run_nvidia_smi

      output = _run_nvidia_smi(['--query-gpu=temperature.gpu', '--format=csv,noheader,nounits', f'--id={self._gpu_info.index}'])
      if output:
        return float(output.strip())
    except (ValueError, TypeError):
      pass
    return 0.0

  def get_gpu_power_draw(self) -> float:
    """Get GPU power draw in watts."""
    if self._gpu_info is None:
      return 0.0

    try:
      from openpilot.system.hardware.nvidia.gpu import _run_nvidia_smi

      output = _run_nvidia_smi(['--query-gpu=power.draw', '--format=csv,noheader,nounits', f'--id={self._gpu_info.index}'])
      if output:
        return float(output.strip())
    except (ValueError, TypeError):
      pass
    return 0.0


class NvidiaGPUThermalZone(ThermalZone):
  """
  Virtual thermal zone for NVIDIA GPU temperature.

  Queries nvidia-smi instead of /sys/class/thermal.
  """

  def __init__(self, gpu_index: int = 0):
    super().__init__(name=f"nvidia_gpu_{gpu_index}", scale=1.0)
    self.gpu_index = gpu_index

  def read(self) -> float:
    """Read GPU temperature via nvidia-smi."""
    try:
      from openpilot.system.hardware.nvidia.gpu import _run_nvidia_smi

      output = _run_nvidia_smi(['--query-gpu=temperature.gpu', '--format=csv,noheader,nounits', f'--id={self.gpu_index}'])
      if output:
        return float(output.strip())
    except (ValueError, TypeError):
      pass
    return 0.0
