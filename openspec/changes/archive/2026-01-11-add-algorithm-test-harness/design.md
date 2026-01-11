## Context

openpilot's control algorithms (lateral control, longitudinal control, lane change logic) are safety-critical and require rigorous testing. The existing infrastructure provides:

- **OpenpilotPrefix**: Test isolation without hardware
- **SimulatedCar**: Synthetic CAN message generation
- **process_replay**: Replay recorded routes against processes
- **Hypothesis tests**: Property-based verification

However, developers experimenting with new algorithms face friction:
1. No standardized way to benchmark algorithm variants
2. No curated scenario library for edge case testing
3. No A/B comparison tooling
4. No unified metrics collection

**Stakeholders**: Algorithm developers, safety engineers, contributors without hardware

## Goals / Non-Goals

**Goals**:
- Enable algorithm experimentation without comma device hardware
- Provide deterministic, reproducible algorithm evaluation
- Standardize metrics for algorithm comparison (latency, tracking error, smoothness)
- Create a curated library of challenging scenarios
- Support both unit-level and integration-level algorithm testing

**Non-Goals**:
- Replace existing CI/CD pipeline (this is supplementary)
- Provide real-time hardware-in-the-loop simulation
- Certify algorithms for production deployment
- Modify safety-critical code paths

## Decisions

### Decision 1: Layered Architecture

The test harness will have three layers:

```
┌─────────────────────────────────────────────────────┐
│            CLI / Reporting (tools/algo_bench.py)    │
├─────────────────────────────────────────────────────┤
│    Scenario Runner (algorithm_harness/runner.py)    │
├─────────────────────────────────────────────────────┤
│  Scenarios │ Metrics │ Vehicle Model │ Algorithm IF │
└─────────────────────────────────────────────────────┘
```

**Rationale**: Separation of concerns allows independent evolution of scenarios, metrics, and runner logic.

### Decision 2: Algorithm Interface Protocol

Algorithms implement a standard interface for harness compatibility:

```python
class AlgorithmInterface(Protocol):
    def reset(self) -> None: ...
    def update(self, state: AlgorithmState) -> AlgorithmOutput: ...

@dataclass
class AlgorithmState:
    v_ego: float              # m/s
    a_ego: float              # m/s^2
    yaw_rate: float           # rad/s
    steering_angle: float     # rad
    curvature: float          # 1/m (path curvature)
    lane_width: float         # m
    # ... extensible

@dataclass
class AlgorithmOutput:
    actuator_output: float    # Algorithm's control output
    saturated: bool           # Output was clipped
    metadata: dict            # Algorithm-specific debug info
```

**Rationale**: Decouples harness from specific algorithm implementations. Existing controllers (LatControlPID, LatControlTorque) can be wrapped to implement this interface.

### Decision 3: Scenario Format

Scenarios are defined as time-series data in Parquet format:

```
scenarios/
├── curves/
│   ├── tight_s_curve.parquet
│   └── highway_exit.parquet
├── lane_changes/
│   └── highway_merge.parquet
└── edge_cases/
    ├── low_speed_maneuver.parquet
    └── icy_conditions.parquet
```

Each scenario contains:
- `timestamp_ns`: Monotonic timestamps
- `v_ego`, `a_ego`, `yaw_rate`, etc.: Vehicle state
- `ground_truth_*`: Expected outputs for validation
- `metadata`: Scenario description, difficulty rating

**Rationale**: Parquet is columnar, efficient, and supports schema evolution. Compatible with pandas/polars for analysis.

**Alternatives considered**:
- JSON: Too verbose for time-series data
- CSV: No schema, poor performance
- Cap'n Proto: Overkill for offline analysis

### Decision 4: Metrics Collection

Standard metrics captured for every algorithm run:

| Metric | Description | Unit |
|--------|-------------|------|
| `tracking_error_rmse` | RMS error vs ground truth | varies |
| `tracking_error_max` | Maximum absolute error | varies |
| `output_smoothness` | Jerk/rate-of-change metric | varies |
| `latency_p50` | Median update latency | ms |
| `latency_p99` | 99th percentile latency | ms |
| `saturation_ratio` | Fraction of time saturated | 0-1 |
| `safety_margin_min` | Minimum safety margin | varies |

**Rationale**: These metrics capture performance, smoothness, and safety characteristics that matter for control algorithms.

### Decision 5: Deterministic Replay Mode

Extend process_replay with `--deterministic` flag:
- Fixes random seeds
- Disables real-time clock dependencies
- Uses monotonic fake time
- Ensures bit-exact reproducibility

**Rationale**: Critical for regression testing and A/B comparisons.

### Decision 6: Vehicle Dynamics Model

Extend `SimulatedCar` with configurable dynamics:

```python
@dataclass
class VehicleDynamicsConfig:
    mass: float = 1500.0           # kg
    wheelbase: float = 2.7         # m
    max_steer_rate: float = 1.5    # rad/s
    steering_ratio: float = 15.0
    tire_stiffness: float = 80000  # N/rad (cornering stiffness)
```

**Rationale**: Different vehicles have different dynamics. Configurable model enables testing across vehicle types without hardware.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Simulation fidelity gap | Document limitations; use recorded scenarios for validation |
| Scenario library maintenance | Start small (10-20 scenarios); community contributions |
| Algorithm interface too restrictive | Keep interface minimal; allow metadata for extensions |
| Performance overhead in CI | Benchmarks run separately from unit tests |

## Migration Plan

1. **Phase 1**: Core harness framework + 5 seed scenarios
2. **Phase 2**: CLI tooling + metrics visualization
3. **Phase 3**: Scenario library expansion + community guidelines
4. **Rollback**: Delete new directories; no modifications to existing code

## Open Questions

1. Should scenario Parquet files be committed to repo or stored externally (LFS)?
2. What's the right granularity for vehicle dynamics models?
3. Should we integrate with existing plotjuggler tooling for visualization?
