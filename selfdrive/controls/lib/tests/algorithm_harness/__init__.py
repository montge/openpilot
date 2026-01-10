"""
Algorithm Test Harness for openpilot control algorithms.

This module provides a standardized framework for testing and benchmarking
control algorithms without requiring comma device hardware.

Key components:
- AlgorithmInterface: Protocol for algorithm implementations
- ScenarioRunner: Execute algorithms against test scenarios
- MetricsCollector: Collect and analyze algorithm performance metrics
- Adapters: Wrap existing openpilot controllers for harness compatibility
- Scenarios: Load, save, and generate test scenarios
"""

from openpilot.selfdrive.controls.lib.tests.algorithm_harness.interface import (
  AlgorithmInterface,
  AlgorithmState,
  AlgorithmOutput,
  LateralAlgorithmState,
  LateralAlgorithmOutput,
  LongitudinalAlgorithmState,
  LongitudinalAlgorithmOutput,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.metrics import (
  MetricsCollector,
  AlgorithmMetrics,
  compare_metrics,
  format_metrics_table,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import (
  Scenario,
  ScenarioRunner,
  ScenarioResult,
  generate_synthetic_scenario,
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.vehicle_dynamics import (
  VehicleDynamicsConfig,
  VehicleType,
  BicycleModel,
  BicycleModelState,
  get_sedan_config,
  get_suv_config,
  get_truck_config,
  get_compact_config,
  get_sports_config,
  get_vehicle_config,
  VEHICLE_PRESETS,
)

__all__ = [
  # Interface
  'AlgorithmInterface',
  'AlgorithmState',
  'AlgorithmOutput',
  'LateralAlgorithmState',
  'LateralAlgorithmOutput',
  'LongitudinalAlgorithmState',
  'LongitudinalAlgorithmOutput',
  # Metrics
  'MetricsCollector',
  'AlgorithmMetrics',
  'compare_metrics',
  'format_metrics_table',
  # Runner
  'Scenario',
  'ScenarioRunner',
  'ScenarioResult',
  'generate_synthetic_scenario',
  # Vehicle Dynamics
  'VehicleDynamicsConfig',
  'VehicleType',
  'BicycleModel',
  'BicycleModelState',
  'get_sedan_config',
  'get_suv_config',
  'get_truck_config',
  'get_compact_config',
  'get_sports_config',
  'get_vehicle_config',
  'VEHICLE_PRESETS',
]
