# Design: add-latcontrol-tests

## Overview

This change adds unit tests for the three lateral control implementations in `selfdrive/controls/lib/`. Each controller inherits from `LatControl` (already tested) and implements the `update()` method with controller-specific logic.

## Test Architecture

### Mock Strategy

Each controller requires mocking several dependencies:

```
CarParams (CP)           - Vehicle configuration
├── steerLimitTimer      - Saturation time limit
├── lateralTuning.*      - Controller-specific tuning
└── brand                - For LatControlAngle Tesla check

CarInterface (CI)        - Car-specific functions
├── torque_from_lateral_accel()   - LatControlTorque
├── lateral_accel_from_torque()   - LatControlTorque
└── get_steer_feedforward_function() - LatControlPID

CarState (CS)            - Current vehicle state
├── vEgo                 - Vehicle speed
├── steeringAngleDeg     - Current steering angle
├── steeringRateDeg      - Steering rate
└── steeringPressed      - Driver override

VehicleModel (VM)        - Physics model
├── calc_curvature()     - Curvature from angle
└── get_steer_from_curvature() - Angle from curvature

params                   - Calibration parameters
├── angleOffsetDeg       - Steering angle offset
└── roll                 - Vehicle roll
```

### Shared Fixtures

Create reusable mock helpers following the pattern from `test_latcontrol.py`:

```python
def create_mock_cp_angle(mocker, brand="honda"):
    """Mock CarParams for LatControlAngle."""
    CP = mocker.MagicMock()
    CP.steerLimitTimer = 4.0
    CP.brand = brand
    return CP

def create_mock_cp_pid(mocker):
    """Mock CarParams for LatControlPID."""
    CP = mocker.MagicMock()
    CP.steerLimitTimer = 4.0
    CP.lateralTuning.pid.kpBP = [0.]
    CP.lateralTuning.pid.kpV = [0.5]
    CP.lateralTuning.pid.kiBP = [0.]
    CP.lateralTuning.pid.kiV = [0.1]
    CP.lateralTuning.pid.kf = 1.0
    return CP

def create_mock_cp_torque(mocker):
    """Mock CarParams for LatControlTorque."""
    CP = mocker.MagicMock()
    CP.steerLimitTimer = 4.0
    CP.lateralTuning.torque.steeringAngleDeadzoneDeg = 0.0
    # as_builder() returns a mutable copy
    CP.lateralTuning.torque.as_builder.return_value = mocker.MagicMock(
        latAccelFactor=1.0,
        latAccelOffset=0.0,
        friction=0.1,
        steeringAngleDeadzoneDeg=0.0
    )
    return CP
```

### Test Categories

For each controller, test:

1. **Initialization** - Correct setup of internal state
2. **Inactive behavior** - Returns safe defaults when `active=False`
3. **Active behavior** - Core control logic
4. **Edge cases** - Boundary conditions, special states
5. **Saturation** - Inherited from base class but triggered by controller

## File Structure

```
selfdrive/controls/lib/tests/
├── test_latcontrol.py        # Existing - base class tests
├── test_latcontrol_angle.py  # New - LatControlAngle tests
├── test_latcontrol_pid.py    # New - LatControlPID tests
└── test_latcontrol_torque.py # New - LatControlTorque tests
```

## Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| `latcontrol_angle.py` | 0% | 95% |
| `latcontrol_pid.py` | 0% | 95% |
| `latcontrol_torque.py` | 0% | 90% |

Note: `latcontrol_torque.py` has more complex dependencies that may require strategic mocking; 90% is realistic given the complexity of the jerk filtering and buffer logic.
