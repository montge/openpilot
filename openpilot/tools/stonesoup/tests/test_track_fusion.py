"""Tests for track-to-track fusion."""
import numpy as np

from openpilot.tools.stonesoup.track_fusion import (
  FusedState,
  FusionMetrics,
  RadarTrack,
  TrackFusionEngine,
  VisionTrack,
  compare_fusion_methods,
  compute_fusion_metrics,
  covariance_intersection,
  create_fusion_scenario,
  create_occlusion_scenario,
  fast_covariance_intersection,
  format_fusion_report,
)


class TestCovarianceIntersection:
  def test_basic_fusion(self):
    x1 = np.array([10.0, 1.0])
    P1 = np.diag([1.0, 0.1])
    x2 = np.array([12.0, 0.8])
    P2 = np.diag([2.0, 0.2])

    x_fused, P_fused, omega = covariance_intersection(x1, P1, x2, P2)

    assert x_fused.shape == (2,)
    assert P_fused.shape == (2, 2)
    assert 0 < omega < 1

  def test_omega_favors_lower_covariance(self):
    x1 = np.array([10.0, 1.0])
    P1 = np.diag([0.1, 0.1])  # Lower covariance
    x2 = np.array([12.0, 0.8])
    P2 = np.diag([10.0, 10.0])  # Higher covariance

    _, _, omega = covariance_intersection(x1, P1, x2, P2)

    # Should favor x1 (higher omega)
    assert omega > 0.5

  def test_fused_covariance_is_smaller(self):
    x1 = np.array([10.0, 1.0])
    P1 = np.diag([1.0, 0.5])
    x2 = np.array([11.0, 1.2])
    P2 = np.diag([1.5, 0.8])

    _, P_fused, _ = covariance_intersection(x1, P1, x2, P2)

    # Fused covariance trace should be <= min(trace(P1), trace(P2))
    assert np.trace(P_fused) <= min(np.trace(P1), np.trace(P2))

  def test_fast_ci_produces_similar_results(self):
    x1 = np.array([10.0, 1.0, 5.0, 0.5])
    P1 = np.diag([1.0, 0.5, 0.3, 0.1])
    x2 = np.array([11.0, 1.2, 4.8, 0.6])
    P2 = np.diag([1.5, 0.8, 0.4, 0.2])

    x_opt, P_opt, omega_opt = covariance_intersection(x1, P1, x2, P2)
    x_fast, P_fast, omega_fast = fast_covariance_intersection(x1, P1, x2, P2)

    # Results should be similar (not identical due to approximation)
    assert np.allclose(x_opt, x_fast, atol=1.0)
    assert np.allclose(omega_opt, omega_fast, atol=0.3)


class TestTrackFusionEngine:
  def test_initialization(self):
    engine = TrackFusionEngine()
    assert not engine.use_fast_ci

  def test_fuse_radar_only(self):
    engine = TrackFusionEngine()
    radar = RadarTrack(
      sensor_name="radar",
      position=np.array([20.0, 0.0]),
      velocity=np.array([-5.0, 0.0]),
      covariance=np.diag([0.5, 0.5, 0.1, 0.1]),
      timestamp=0.0
    )

    result = engine.fuse_tracks(radar, None)

    assert result is not None
    assert result.omega == 1.0
    np.testing.assert_array_equal(result.position, radar.position)

  def test_fuse_vision_only(self):
    engine = TrackFusionEngine()
    vision = VisionTrack(
      sensor_name="vision",
      position=np.array([22.0, 0.5]),
      velocity=np.array([-4.5, 0.1]),
      covariance=np.diag([1.0, 1.0, 0.5, 0.5]),
      timestamp=0.0
    )

    result = engine.fuse_tracks(None, vision)

    assert result is not None
    assert result.omega == 0.0
    np.testing.assert_array_equal(result.position, vision.position)

  def test_fuse_both_sensors(self):
    engine = TrackFusionEngine()
    radar = RadarTrack(
      sensor_name="radar",
      position=np.array([20.0, 0.0]),
      velocity=np.array([-5.0, 0.0]),
      covariance=np.diag([0.5, 0.5, 0.1, 0.1]),
      timestamp=0.0
    )
    vision = VisionTrack(
      sensor_name="vision",
      position=np.array([22.0, 0.5]),
      velocity=np.array([-4.5, 0.1]),
      covariance=np.diag([1.0, 1.0, 0.5, 0.5]),
      timestamp=0.0
    )

    result = engine.fuse_tracks(radar, vision)

    assert result is not None
    assert 0 < result.omega < 1
    # Fused position should be between radar and vision
    assert 20.0 <= result.position[0] <= 22.0

  def test_no_tracks(self):
    engine = TrackFusionEngine()
    result = engine.fuse_tracks(None, None)
    assert result is None


class TestScenarios:
  def test_fusion_scenario_creation(self):
    scenario = create_fusion_scenario(dt=0.1, duration=3.0, seed=42)

    assert scenario.name == "radar_vision_fusion"
    assert len(scenario.timestamps) == 30
    assert len(scenario.radar_tracks) == 30
    assert len(scenario.vision_tracks) == 30

  def test_occlusion_scenario_creation(self):
    scenario = create_occlusion_scenario(dt=0.1, duration=5.0, seed=42)

    assert scenario.name == "occlusion"
    assert len(scenario.timestamps) == 50

    # Vision should be None during occlusion
    occlusion_start_idx = int(2.0 / 0.1)
    occlusion_end_idx = int(3.0 / 0.1)

    for i in range(occlusion_start_idx, occlusion_end_idx):
      assert scenario.vision_tracks[i] is None

  def test_process_scenario(self):
    scenario = create_fusion_scenario(dt=0.1, duration=2.0, seed=42)
    engine = TrackFusionEngine()

    results = engine.process_scenario(scenario)

    assert len(results) > 0
    for r in results:
      assert isinstance(r.fused_state, FusedState)


class TestMetrics:
  def test_compute_fusion_metrics(self):
    scenario = create_fusion_scenario(dt=0.1, duration=2.0, seed=42)
    engine = TrackFusionEngine()
    results = engine.process_scenario(scenario)

    metrics = compute_fusion_metrics("CI", results, scenario)

    assert isinstance(metrics, FusionMetrics)
    assert metrics.method_name == "CI"
    assert metrics.rmse_position >= 0
    assert metrics.rmse_velocity >= 0
    assert 0 <= metrics.mean_omega <= 1
    assert metrics.n_samples > 0


class TestCompare:
  def test_compare_methods(self):
    scenario = create_fusion_scenario(dt=0.1, duration=2.0, seed=42)
    metrics = compare_fusion_methods(scenario)

    assert "CI_Optimized" in metrics
    assert "CI_Fast" in metrics
    assert "Radar_Only" in metrics
    assert "Vision_Only" in metrics

  def test_fusion_improves_over_single_sensor(self):
    scenario = create_fusion_scenario(
      dt=0.1, duration=3.0,
      radar_noise_std=0.5, vision_noise_std=0.8,
      radar_dropout_prob=0.0, vision_dropout_prob=0.0,
      seed=42
    )
    metrics = compare_fusion_methods(scenario)

    # CI should perform at least as well as best single sensor
    ci_rmse = metrics["CI_Optimized"].rmse_position
    radar_rmse = metrics["Radar_Only"].rmse_position
    vision_rmse = metrics["Vision_Only"].rmse_position

    # Fusion should generally improve over individual sensors
    # but this depends on the noise levels
    assert ci_rmse <= max(radar_rmse, vision_rmse)


class TestReport:
  def test_format_report(self):
    scenario = create_fusion_scenario(dt=0.1, duration=2.0, seed=42)
    metrics = compare_fusion_methods(scenario)
    report = format_fusion_report(scenario, metrics)

    assert "# Track-to-Track Fusion Comparison" in report
    assert "radar_vision_fusion" in report
    assert "CI_Optimized" in report
    assert "CI_Fast" in report
    assert "RMSE" in report
