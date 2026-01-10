"""
Algorithm Test Harness for openpilot control algorithms.

This module provides a standardized framework for testing and benchmarking
control algorithms without requiring comma device hardware.

Key components:
- AlgorithmInterface: Protocol for algorithm implementations
- ScenarioRunner: Execute algorithms against test scenarios
- MetricsCollector: Collect and analyze algorithm performance metrics
- Adapters: Wrap existing openpilot controllers for harness compatibility
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
)
from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import (
  ScenarioRunner,
  ScenarioResult,
)

__all__ = [
  'AlgorithmInterface',
  'AlgorithmState',
  'AlgorithmOutput',
  'LateralAlgorithmState',
  'LateralAlgorithmOutput',
  'LongitudinalAlgorithmState',
  'LongitudinalAlgorithmOutput',
  'MetricsCollector',
  'AlgorithmMetrics',
  'ScenarioRunner',
  'ScenarioResult',
]
