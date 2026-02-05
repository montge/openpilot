"""Tests for filter comparison harness."""
import numpy as np

from openpilot.tools.stonesoup.comparison import (
  ComparisonMetrics,
  KF1DWrapper,
  ParticleFilterWrapper,
  StoneSoupKalmanWrapper,
  compare_filters,
  compute_metrics,
  create_acceleration_scenario,
  create_constant_velocity_scenario,
  format_comparison_report,
  run_filter_on_scenario,
)


class TestKF1DWrapper:
  def test_initialization(self):
    kf = KF1DWrapper(dt=0.05)
    state = kf.get_state()
    assert len(state) == 2
    assert state[0] == 0.0
    assert state[1] == 0.0

  def test_update(self):
    kf = KF1DWrapper(dt=0.05)
    state = kf.update(10.0)
    assert state[0] > 0.0  # Position should move toward measurement

  def test_reset(self):
    kf = KF1DWrapper(dt=0.05)
    kf.update(100.0)
    kf.reset(np.array([50.0, -5.0]))
    state = kf.get_state()
    assert state[0] == 50.0
    assert state[1] == -5.0


class TestStoneSoupKalmanWrapper:
  def test_kalman_filter(self):
    kf = StoneSoupKalmanWrapper(dt=0.05, filter_type="kalman")
    state = kf.update(10.0)
    assert len(state) == 2

  def test_ekf(self):
    kf = StoneSoupKalmanWrapper(dt=0.05, filter_type="ekf")
    state = kf.update(10.0)
    assert len(state) == 2

  def test_ukf(self):
    kf = StoneSoupKalmanWrapper(dt=0.05, filter_type="ukf")
    state = kf.update(10.0)
    assert len(state) == 2

  def test_ckf(self):
    kf = StoneSoupKalmanWrapper(dt=0.05, filter_type="ckf")
    state = kf.update(10.0)
    assert len(state) == 2

  def test_reset(self):
    kf = StoneSoupKalmanWrapper(dt=0.05, filter_type="kalman")
    kf.update(100.0)
    kf.reset(np.array([50.0, -5.0]))
    state = kf.get_state()
    assert state[0] == 50.0
    assert state[1] == -5.0


class TestParticleFilterWrapper:
  def test_initialization(self):
    pf = ParticleFilterWrapper(dt=0.05, n_particles=100)
    state = pf.get_state()
    assert len(state) == 2

  def test_update(self):
    pf = ParticleFilterWrapper(dt=0.05, n_particles=100)
    state = pf.update(10.0)
    assert len(state) == 2


class TestScenarios:
  def test_constant_velocity_scenario(self):
    scenario = create_constant_velocity_scenario(
      dt=0.05, duration=5.0, initial_position=50.0, velocity=-5.0
    )
    assert scenario.name == "constant_velocity"
    assert len(scenario.timestamps) == 100
    assert scenario.ground_truth_position[0] == 50.0
    assert np.allclose(scenario.ground_truth_velocity, -5.0)

  def test_acceleration_scenario(self):
    scenario = create_acceleration_scenario(
      dt=0.05, duration=5.0, initial_position=50.0,
      initial_velocity=-5.0, acceleration=-2.0
    )
    assert scenario.name == "constant_acceleration"
    assert len(scenario.timestamps) == 100
    # Velocity should change over time
    assert scenario.ground_truth_velocity[-1] < scenario.ground_truth_velocity[0]


class TestRunFilterOnScenario:
  def test_kf1d_on_constant_velocity(self):
    scenario = create_constant_velocity_scenario(dt=0.05, duration=2.0)
    kf = KF1DWrapper(dt=scenario.dt)
    results = run_filter_on_scenario(kf, scenario)

    assert len(results) == len(scenario.timestamps)
    assert results[0].ground_truth is not None
    assert results[-1].state is not None


class TestComputeMetrics:
  def test_metrics_computation(self):
    scenario = create_constant_velocity_scenario(dt=0.05, duration=2.0)
    kf = KF1DWrapper(dt=scenario.dt)
    results = run_filter_on_scenario(kf, scenario)
    metrics = compute_metrics("KF1D", results)

    assert isinstance(metrics, ComparisonMetrics)
    assert metrics.filter_name == "KF1D"
    assert metrics.rmse_position >= 0
    assert metrics.rmse_velocity >= 0
    assert metrics.n_samples > 0


class TestCompareFilters:
  def test_compare_all_filters(self):
    scenario = create_constant_velocity_scenario(dt=0.05, duration=2.0)
    metrics = compare_filters(scenario)

    assert "KF1D" in metrics
    assert "StoneSoup_Kalman" in metrics
    assert "StoneSoup_EKF" in metrics
    assert "StoneSoup_UKF" in metrics
    assert "StoneSoup_CKF" in metrics
    assert "StoneSoup_Particle" in metrics

  def test_compare_subset_of_filters(self):
    scenario = create_constant_velocity_scenario(dt=0.05, duration=2.0)
    filter_configs = {
      "KF1D": KF1DWrapper(dt=scenario.dt),
      "Kalman": StoneSoupKalmanWrapper(dt=scenario.dt, filter_type="kalman"),
    }
    metrics = compare_filters(scenario, filter_configs)

    assert len(metrics) == 2
    assert "KF1D" in metrics
    assert "Kalman" in metrics


class TestFormatReport:
  def test_format_comparison_report(self):
    scenario = create_constant_velocity_scenario(dt=0.05, duration=2.0)
    filter_configs = {
      "KF1D": KF1DWrapper(dt=scenario.dt),
      "Kalman": StoneSoupKalmanWrapper(dt=scenario.dt, filter_type="kalman"),
    }
    metrics = compare_filters(scenario, filter_configs)
    report = format_comparison_report(scenario, metrics)

    assert "# Filter Comparison Report" in report
    assert "constant_velocity" in report
    assert "KF1D" in report
    assert "Kalman" in report
    assert "RMSE Pos" in report
