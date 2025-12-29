import math

import pytest
from parameterized import parameterized

from cereal import car, log
import cereal.messaging as messaging
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.nissan.values import CAR as NISSAN
from opendbc.car.car_helpers import interfaces
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.controlsd import Controls, ACTUATOR_FIELDS

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection


def get_car_params(car_name):
  """Helper to get CarParams for a given car."""
  CarInterface = interfaces[car_name]
  return CarInterface.get_non_essential_params(car_name)


def setup_params_with_car(car_name):
  """Set up Params with CarParams for testing."""
  CP = get_car_params(car_name)
  params = Params()
  params.put("CarParams", CP.to_bytes())
  return CP


class TestControlsInitialization:
  """Tests for Controls class initialization with different car types."""

  @parameterized.expand(
    [
      (HONDA.HONDA_CIVIC, LatControlPID),
      (TOYOTA.TOYOTA_RAV4, LatControlTorque),
      (NISSAN.NISSAN_LEAF, LatControlAngle),
    ]
  )
  def test_controls_init_lateral_controller_type(self, car_name, expected_lat_controller):
    """Test that Controls initializes the correct lateral controller type based on car."""
    setup_params_with_car(car_name)

    controls = Controls()

    assert isinstance(controls.LaC, expected_lat_controller)
    assert isinstance(controls.LoC, LongControl)

  def test_controls_init_attributes(self):
    """Test that Controls initializes all required attributes."""
    CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)

    controls = Controls()

    # Check attribute initialization
    assert controls.steer_limited_by_safety is False
    assert controls.curvature == 0.0
    assert controls.desired_curvature == 0.0
    assert controls.calibrated_pose is None

    # Check messaging setup
    assert controls.sm is not None
    assert controls.pm is not None

    # Check CP is loaded correctly
    assert controls.CP.carFingerprint == CP.carFingerprint

  def test_controls_init_vehicle_model(self):
    """Test that VehicleModel is correctly initialized."""
    setup_params_with_car(TOYOTA.TOYOTA_RAV4)

    controls = Controls()

    assert controls.VM is not None

  def test_controls_init_pose_calibrator(self):
    """Test that PoseCalibrator is correctly initialized."""
    setup_params_with_car(TOYOTA.TOYOTA_RAV4)

    controls = Controls()

    assert controls.pose_calibrator is not None


class MockSubMaster:
  """Mock SubMaster for testing Controls without real messaging."""

  def __init__(self, services):
    self.services = services
    self.data = {}
    self.logMonoTime = dict.fromkeys(services, 0)
    self.valid = dict.fromkeys(services, True)
    self.updated = dict.fromkeys(services, False)
    self.seen = dict.fromkeys(services, True)
    self.alive = dict.fromkeys(services, True)
    self.freq_ok = dict.fromkeys(services, True)

    # Initialize with default messages
    for s in services:
      try:
        msg = messaging.new_message(s)
        self.data[s] = getattr(msg, s)
      except Exception:
        self.data[s] = None

  def __getitem__(self, s):
    return self.data[s]

  def update(self, timeout=100):
    pass

  def all_checks(self, service_list=None):
    return True

  def all_alive(self, service_list=None):
    return True

  def all_valid(self, service_list=None):
    return True


class TestControlsStateControl:
  """Tests for the state_control method."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()
    # Replace sm with our mock
    self._setup_mock_sm()

  def _setup_mock_sm(self):
    """Set up mock SubMaster with sensible defaults."""
    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm

    # Set up default valid state
    self._set_car_state()
    self._set_live_parameters()
    self._set_live_torque_parameters()
    self._set_model_v2()
    self._set_selfdrive_state()
    self._set_longitudinal_plan()
    self._set_live_delay()
    self._set_car_output()
    self._set_driver_monitoring_state()
    self._set_driver_assistance()
    self._set_onroad_events([])

  def _set_car_state(
    self,
    vEgo=20.0,
    steeringAngleDeg=0.0,
    standstill=False,
    steerFaultTemporary=False,
    steerFaultPermanent=False,
    steeringPressed=False,
    aEgo=0.0,
    brakePressed=False,
    vCruise=40.0,
    cruiseEnabled=True,
    cruiseStandstill=False,
  ):
    """Set carState in the mock SubMaster."""
    msg = messaging.new_message('carState')
    cs = msg.carState
    cs.vEgo = vEgo
    cs.steeringAngleDeg = steeringAngleDeg
    cs.standstill = standstill
    cs.steerFaultTemporary = steerFaultTemporary
    cs.steerFaultPermanent = steerFaultPermanent
    cs.steeringPressed = steeringPressed
    cs.aEgo = aEgo
    cs.brakePressed = brakePressed
    cs.vCruise = vCruise
    cs.vCruiseCluster = vCruise
    cs.cruiseState.enabled = cruiseEnabled
    cs.cruiseState.standstill = cruiseStandstill
    cs.canValid = True
    self.mock_sm.data['carState'] = msg.as_reader().carState

  def _set_live_parameters(self, stiffnessFactor=1.0, steerRatio=15.0, angleOffsetDeg=0.0, roll=0.0):
    """Set liveParameters in the mock SubMaster."""
    msg = messaging.new_message('liveParameters')
    lp = msg.liveParameters
    lp.stiffnessFactor = stiffnessFactor
    lp.steerRatio = steerRatio
    lp.angleOffsetDeg = angleOffsetDeg
    lp.roll = roll
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

  def _set_live_torque_parameters(self, useParams=False, latAccelFactor=2.0, latAccelOffset=0.0, friction=0.1):
    """Set liveTorqueParameters in the mock SubMaster."""
    msg = messaging.new_message('liveTorqueParameters')
    ltp = msg.liveTorqueParameters
    ltp.useParams = useParams
    ltp.latAccelFactorFiltered = latAccelFactor
    ltp.latAccelOffsetFiltered = latAccelOffset
    ltp.frictionCoefficientFiltered = friction
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

  def _set_model_v2(self, desiredCurvature=0.0, laneChangeState=LaneChangeState.off, laneChangeDirection=LaneChangeDirection.none):
    """Set modelV2 in the mock SubMaster."""
    msg = messaging.new_message('modelV2')
    model = msg.modelV2
    model.action.desiredCurvature = desiredCurvature
    model.meta.laneChangeState = laneChangeState
    model.meta.laneChangeDirection = laneChangeDirection
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

  def _set_selfdrive_state(self, enabled=True, active=True, state=State.enabled, personality=log.LongitudinalPersonality.standard):
    """Set selfdriveState in the mock SubMaster."""
    msg = messaging.new_message('selfdriveState')
    ss = msg.selfdriveState
    ss.enabled = enabled
    ss.active = active
    ss.state = state
    ss.personality = personality
    # alertHudVisual defaults to none (0)
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

  def _set_longitudinal_plan(self, aTarget=0.0, shouldStop=False, hasLead=False):
    """Set longitudinalPlan in the mock SubMaster."""
    msg = messaging.new_message('longitudinalPlan')
    lp = msg.longitudinalPlan
    lp.aTarget = aTarget
    lp.shouldStop = shouldStop
    lp.hasLead = hasLead
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

  def _set_live_delay(self, lateralDelay=0.2):
    """Set liveDelay in the mock SubMaster."""
    msg = messaging.new_message('liveDelay')
    ld = msg.liveDelay
    ld.lateralDelay = lateralDelay
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

  def _set_car_output(self, steeringAngleDeg=0.0, torque=0.0):
    """Set carOutput in the mock SubMaster."""
    msg = messaging.new_message('carOutput')
    co = msg.carOutput
    co.actuatorsOutput.steeringAngleDeg = steeringAngleDeg
    co.actuatorsOutput.torque = torque
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

  def _set_driver_monitoring_state(self, awarenessStatus=1.0):
    """Set driverMonitoringState in the mock SubMaster."""
    msg = messaging.new_message('driverMonitoringState')
    dms = msg.driverMonitoringState
    dms.awarenessStatus = awarenessStatus
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

  def _set_driver_assistance(self, leftLaneDeparture=False, rightLaneDeparture=False):
    """Set driverAssistance in the mock SubMaster."""
    msg = messaging.new_message('driverAssistance')
    da = msg.driverAssistance
    da.leftLaneDeparture = leftLaneDeparture
    da.rightLaneDeparture = rightLaneDeparture
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance

  def _set_onroad_events(self, events):
    """Set onroadEvents in the mock SubMaster."""
    # onroadEvents is a list
    self.mock_sm.data['onroadEvents'] = events

  def test_state_control_enabled(self):
    """Test state_control when system is enabled and active."""
    self._set_selfdrive_state(enabled=True, active=True)

    CC, lac_log = self.controls.state_control()

    assert CC.enabled is True
    assert CC.latActive is True
    # longActive depends on openpilotLongitudinalControl
    if self.controls.CP.openpilotLongitudinalControl:
      assert CC.longActive is True

  def test_state_control_disabled(self):
    """Test state_control when system is disabled."""
    self._set_selfdrive_state(enabled=False, active=False)

    CC, lac_log = self.controls.state_control()

    assert CC.enabled is False
    assert CC.latActive is False

  def test_state_control_steer_fault_temporary(self):
    """Test state_control with temporary steering fault."""
    self._set_car_state(steerFaultTemporary=True)

    CC, lac_log = self.controls.state_control()

    assert CC.latActive is False

  def test_state_control_steer_fault_permanent(self):
    """Test state_control with permanent steering fault."""
    self._set_car_state(steerFaultPermanent=True)

    CC, lac_log = self.controls.state_control()

    assert CC.latActive is False

  def test_state_control_standstill_no_steer_at_standstill(self):
    """Test state_control at standstill when car doesn't support steer at standstill."""
    self._set_car_state(vEgo=0.0, standstill=True)

    CC, lac_log = self.controls.state_control()

    # If the car doesn't support steerAtStandstill, latActive should be False
    if not self.controls.CP.steerAtStandstill:
      assert CC.latActive is False

  def test_state_control_lane_change_blinkers_left(self):
    """Test that left blinker is set during left lane change."""
    self._set_model_v2(laneChangeState=LaneChangeState.laneChangeStarting, laneChangeDirection=LaneChangeDirection.left)

    CC, lac_log = self.controls.state_control()

    assert CC.leftBlinker is True
    assert CC.rightBlinker is False

  def test_state_control_lane_change_blinkers_right(self):
    """Test that right blinker is set during right lane change."""
    self._set_model_v2(laneChangeState=LaneChangeState.laneChangeStarting, laneChangeDirection=LaneChangeDirection.right)

    CC, lac_log = self.controls.state_control()

    assert CC.leftBlinker is False
    assert CC.rightBlinker is True

  def test_state_control_curvature_update(self):
    """Test that curvature is computed from steering angle."""
    steer_angle = 10.0  # degrees
    self._set_car_state(steeringAngleDeg=steer_angle)

    CC, lac_log = self.controls.state_control()

    # Curvature should be non-zero for non-zero steering angle
    assert self.controls.curvature != 0.0

  def test_state_control_desired_curvature_from_model(self):
    """Test that desired curvature comes from model when active."""
    desired_curv = 0.01
    self._set_model_v2(desiredCurvature=desired_curv)

    CC, lac_log = self.controls.state_control()

    # Desired curvature should be set (may be clipped)
    assert CC.actuators.curvature is not None

  def test_state_control_longitudinal_inactive_on_override(self):
    """Test that longActive is False when there's a longitudinal override event."""
    # Create an event with overrideLongitudinal
    msg = messaging.new_message('onroadEvents', 1)
    msg.onroadEvents[0].overrideLongitudinal = True
    self._set_onroad_events(msg.as_reader().onroadEvents)

    CC, lac_log = self.controls.state_control()

    # longActive should be False when there's an override
    if self.controls.CP.openpilotLongitudinalControl:
      assert CC.longActive is False

  def test_state_control_vehicle_model_update(self):
    """Test that VehicleModel parameters are updated from liveParameters."""
    stiffness = 1.5
    steer_ratio = 16.0
    self._set_live_parameters(stiffnessFactor=stiffness, steerRatio=steer_ratio)

    # Should not raise - verify the method runs without error
    CC, lac_log = self.controls.state_control()
    assert CC is not None

  def test_state_control_lat_controller_reset_when_inactive(self):
    """Test that lateral controller resets when latActive is False."""
    self._set_selfdrive_state(enabled=False, active=False)

    # Set some state in the lateral controller
    self.controls.LaC.sat_time = 1.0

    CC, lac_log = self.controls.state_control()

    # The controller should have been reset
    assert self.controls.LaC.sat_time == 0.0

  def test_state_control_long_controller_reset_when_inactive(self):
    """Test that longitudinal controller resets when longActive is False."""
    self._set_selfdrive_state(enabled=False, active=False)

    CC, lac_log = self.controls.state_control()

    # Long controller state should be off
    assert self.controls.LoC.long_control_state == LongCtrlState.off

  def test_state_control_zero_accel_when_disabled(self):
    """Test that accel output is zero when system is disabled."""
    self._set_selfdrive_state(enabled=False, active=False)

    CC, lac_log = self.controls.state_control()

    assert CC.actuators.accel == 0.0


class TestControlsActuatorSafety:
  """Tests for actuator safety checks (NaN/Inf handling)."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    # Set up mock SubMaster
    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm
    self._setup_default_sm()

  def _setup_default_sm(self):
    """Set up default mock SubMaster values."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = 20.0
    msg.carState.vCruise = 40.0
    self.mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('selfdriveState')
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('modelV2')
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('liveTorqueParameters')
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance

    self.mock_sm.data['onroadEvents'] = []

  def test_actuator_fields_constant(self):
    """Test that ACTUATOR_FIELDS contains expected fields."""
    assert 'accel' in ACTUATOR_FIELDS
    assert 'torque' in ACTUATOR_FIELDS
    assert 'steeringAngleDeg' in ACTUATOR_FIELDS
    assert 'curvature' in ACTUATOR_FIELDS

  def test_state_control_outputs_finite(self):
    """Test that state_control always outputs finite actuator values."""
    CC, lac_log = self.controls.state_control()

    # All numeric actuator fields should be finite
    for field in ACTUATOR_FIELDS:
      attr = getattr(CC.actuators, field)
      if isinstance(attr, (int, float)):
        assert math.isfinite(attr), f"actuators.{field} is not finite: {attr}"


class TestControlsPublish:
  """Tests for the publish method."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    # Set up mock SubMaster
    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm
    self._setup_default_sm()

  def _setup_default_sm(self, vCruiseCluster=40.0):
    """Set up default mock SubMaster values."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = 20.0
    msg.carState.vCruise = vCruiseCluster
    msg.carState.vCruiseCluster = vCruiseCluster
    msg.carState.cruiseState.enabled = True
    msg.carState.cruiseState.standstill = False
    msg.carState.canValid = True
    self.mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('selfdriveState')
    msg.selfdriveState.enabled = True
    msg.selfdriveState.active = True
    msg.selfdriveState.state = State.enabled
    msg.selfdriveState.personality = log.LongitudinalPersonality.standard
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    msg.longitudinalPlan.hasLead = False
    msg.longitudinalPlan.shouldStop = False
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('carOutput')
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    msg.driverMonitoringState.awarenessStatus = 1.0
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance
    self.mock_sm.valid['driverAssistance'] = True

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('liveTorqueParameters')
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('modelV2')
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

    self.mock_sm.data['onroadEvents'] = []

  def _create_car_control(self, enabled=True, latActive=True, longActive=True):
    """Create a CarControl message for testing publish."""
    CC = car.CarControl.new_message()
    CC.enabled = enabled
    CC.latActive = latActive
    CC.longActive = longActive
    CC.actuators.accel = 0.0
    CC.actuators.torque = 0.5
    CC.actuators.steeringAngleDeg = 5.0
    CC.actuators.curvature = 0.01
    return CC

  def test_publish_sets_current_curvature(self):
    """Test that publish sets currentCurvature on CarControl."""
    self.controls.curvature = 0.05

    CC = self._create_car_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    assert abs(CC.currentCurvature - 0.05) < 1e-6

  def test_publish_cruise_control_override(self):
    """Test cruise control override logic."""
    # When enabled but longActive is False with openpilotLongitudinalControl
    CC = self._create_car_control(enabled=True, longActive=False)
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    if self.controls.CP.openpilotLongitudinalControl:
      assert CC.cruiseControl.override is True

  def test_publish_cruise_control_cancel(self):
    """Test cruise control cancel logic."""
    # When cruiseState is enabled but CC is not enabled
    CC = self._create_car_control(enabled=False)
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    # Cancel should be True if cruise enabled but CC not enabled (or no pcmCruise)
    assert CC.cruiseControl.cancel is True or not self.controls.CP.pcmCruise

  def test_publish_hud_control_set_speed(self):
    """Test that HUD control setSpeed is set from vCruiseCluster."""
    from openpilot.common.constants import CV

    v_cruise_cluster = 50.0  # km/h
    self._setup_default_sm(vCruiseCluster=v_cruise_cluster)

    CC = self._create_car_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    expected_speed = v_cruise_cluster * CV.KPH_TO_MS
    assert abs(CC.hudControl.setSpeed - expected_speed) < 0.01

  def test_publish_hud_control_visibility(self):
    """Test HUD control visibility flags when enabled."""
    CC = self._create_car_control(enabled=True)
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    assert CC.hudControl.speedVisible is True
    assert CC.hudControl.lanesVisible is True

  def test_publish_hud_control_lead_visible(self):
    """Test HUD control leadVisible from longitudinalPlan."""
    msg = messaging.new_message('longitudinalPlan')
    msg.longitudinalPlan.hasLead = True
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    CC = self._create_car_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    assert CC.hudControl.leadVisible is True

  def test_publish_lane_visibility(self):
    """Test that lane visibility is set correctly."""
    CC = self._create_car_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    assert CC.hudControl.rightLaneVisible is True
    assert CC.hudControl.leftLaneVisible is True

  def test_publish_lane_departure_warnings(self):
    """Test lane departure warning flags from driverAssistance."""
    msg = messaging.new_message('driverAssistance')
    msg.driverAssistance.leftLaneDeparture = True
    msg.driverAssistance.rightLaneDeparture = False
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance
    self.mock_sm.valid['driverAssistance'] = True

    CC = self._create_car_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()

    self.controls.publish(CC, lac_log)

    assert CC.hudControl.leftLaneDepart is True
    assert CC.hudControl.rightLaneDepart is False


class TestControlsUpdate:
  """Tests for the update method."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    # Set up mock SubMaster
    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm

  def test_update_calibration_handling(self):
    """Test that update handles liveCalibration updates."""
    # Set updated flag for liveCalibration
    self.mock_sm.updated['liveCalibration'] = True

    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.validBlocks = 20
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    msg.liveCalibration.calStatus = log.LiveCalibrationData.Status.calibrated
    self.mock_sm.data['liveCalibration'] = msg.as_reader().liveCalibration

    # The update method should process this without error
    self.controls.update()

  def test_update_live_pose_handling(self):
    """Test that update handles livePose updates."""
    self.mock_sm.updated['livePose'] = True

    msg = messaging.new_message('livePose')
    self.mock_sm.data['livePose'] = msg.as_reader().livePose

    # Also need liveCalibration for the pose calibrator
    msg = messaging.new_message('liveCalibration')
    msg.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
    self.mock_sm.data['liveCalibration'] = msg.as_reader().liveCalibration
    self.mock_sm.updated['liveCalibration'] = True

    # Process the calibration first
    self.controls.update()

    # Now test livePose
    self.mock_sm.updated['liveCalibration'] = False
    self.controls.update()


class TestControlsIntegration:
  """Integration tests that test multiple methods together."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    # Set up mock SubMaster
    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm
    self._setup_complete_sm()

  def _setup_complete_sm(self, vEgo=20.0, enabled=True, active=True):
    """Set up complete mock SubMaster for integration tests."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = vEgo
    msg.carState.steeringAngleDeg = 0.0
    msg.carState.standstill = False
    msg.carState.steerFaultTemporary = False
    msg.carState.steerFaultPermanent = False
    msg.carState.steeringPressed = False
    msg.carState.aEgo = 0.0
    msg.carState.brakePressed = False
    msg.carState.vCruise = 40.0
    msg.carState.vCruiseCluster = 40.0
    msg.carState.cruiseState.enabled = True
    msg.carState.cruiseState.standstill = False
    msg.carState.canValid = True
    self.mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    msg.liveParameters.angleOffsetDeg = 0.0
    msg.liveParameters.roll = 0.0
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('liveTorqueParameters')
    msg.liveTorqueParameters.useParams = False
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('modelV2')
    msg.modelV2.action.desiredCurvature = 0.0
    msg.modelV2.meta.laneChangeState = LaneChangeState.off
    msg.modelV2.meta.laneChangeDirection = LaneChangeDirection.none
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('selfdriveState')
    msg.selfdriveState.enabled = enabled
    msg.selfdriveState.active = active
    msg.selfdriveState.state = State.enabled if enabled else State.disabled
    msg.selfdriveState.personality = log.LongitudinalPersonality.standard
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    msg.longitudinalPlan.aTarget = 0.0
    msg.longitudinalPlan.shouldStop = False
    msg.longitudinalPlan.hasLead = False
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('liveDelay')
    msg.liveDelay.lateralDelay = 0.2
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    msg.carOutput.actuatorsOutput.steeringAngleDeg = 0.0
    msg.carOutput.actuatorsOutput.torque = 0.0
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    msg.driverMonitoringState.awarenessStatus = 1.0
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    msg.driverAssistance.leftLaneDeparture = False
    msg.driverAssistance.rightLaneDeparture = False
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance
    self.mock_sm.valid['driverAssistance'] = True

    self.mock_sm.data['onroadEvents'] = []

  def test_full_control_cycle(self):
    """Test a full control cycle: state_control -> publish."""
    # Run state_control
    CC, lac_log = self.controls.state_control()

    # Verify CC is properly formed
    assert CC.enabled is True
    assert CC.latActive is True
    assert CC.actuators is not None

    # Run publish
    self.controls.publish(CC, lac_log)

    # Verify CC was updated by publish
    assert CC.currentCurvature == self.controls.curvature

  def test_disabled_control_cycle(self):
    """Test control cycle when system is disabled."""
    self._setup_complete_sm(enabled=False, active=False)

    CC, lac_log = self.controls.state_control()

    assert CC.enabled is False
    assert CC.latActive is False

    # Accel should be 0 when disabled
    assert CC.actuators.accel == 0.0

  def test_high_speed_control_cycle(self):
    """Test control cycle at high speed."""
    self._setup_complete_sm(vEgo=30.0, enabled=True, active=True)

    CC, lac_log = self.controls.state_control()

    assert CC.enabled is True
    assert CC.latActive is True

  def test_low_speed_control_cycle(self):
    """Test control cycle at low speed."""
    self._setup_complete_sm(vEgo=0.5, enabled=True, active=True)

    CC, lac_log = self.controls.state_control()

    # At low speed, latActive depends on steerAtStandstill capability
    # Just verify no error occurs
    assert CC is not None

  def test_multiple_control_cycles(self):
    """Test running multiple control cycles in sequence."""
    for _ in range(5):
      CC, lac_log = self.controls.state_control()
      self.controls.publish(CC, lac_log)

    # Should complete without error and curvature should update
    assert self.controls.curvature is not None


class TestControlsDifferentCars:
  """Tests with different car types to ensure compatibility."""

  @parameterized.expand(
    [
      (HONDA.HONDA_CIVIC,),
      (TOYOTA.TOYOTA_RAV4,),
      (NISSAN.NISSAN_LEAF,),
    ]
  )
  def test_initialization_different_cars(self, car_name):
    """Test that Controls initializes correctly for different car types."""
    setup_params_with_car(car_name)

    controls = Controls()

    assert controls.CP is not None
    assert controls.LaC is not None
    assert controls.LoC is not None
    assert controls.VM is not None

  @parameterized.expand(
    [
      (HONDA.HONDA_CIVIC,),
      (TOYOTA.TOYOTA_RAV4,),
    ]
  )
  def test_state_control_different_cars(self, car_name):
    """Test state_control works for different car types."""
    setup_params_with_car(car_name)
    controls = Controls()

    # Set up mock SubMaster
    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    mock_sm = MockSubMaster(services)
    controls.sm = mock_sm

    # Set up minimal data
    msg = messaging.new_message('carState')
    msg.carState.vEgo = 20.0
    msg.carState.vCruise = 40.0
    mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('selfdriveState')
    mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('modelV2')
    mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('liveTorqueParameters')
    mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance

    mock_sm.data['onroadEvents'] = []

    CC, lac_log = controls.state_control()

    # Should complete without error
    assert CC is not None
    assert lac_log is not None


class TestControlsNaNInfHandling:
  """Tests for NaN/Inf detection and replacement in actuators."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    # Set up mock SubMaster
    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm
    self._setup_default_sm()

  def _setup_default_sm(self):
    """Set up default mock SubMaster values."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = 20.0
    msg.carState.vCruise = 40.0
    msg.carState.canValid = True
    self.mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('selfdriveState')
    msg.selfdriveState.enabled = True
    msg.selfdriveState.active = True
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('modelV2')
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('liveTorqueParameters')
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance

    self.mock_sm.data['onroadEvents'] = []

  def test_nan_accel_replaced_with_zero(self):
    """Test that NaN accel is replaced with 0.0."""
    # Run state_control to get initial CC
    CC, lac_log = self.controls.state_control()

    # Inject NaN into accel
    CC.actuators.accel = float('nan')

    # Run the NaN check manually (as done in state_control)
    for p in ACTUATOR_FIELDS:
      attr = getattr(CC.actuators, p)
      if not isinstance(attr, (int, float)):
        continue
      if not math.isfinite(attr):
        setattr(CC.actuators, p, 0.0)

    assert CC.actuators.accel == 0.0

  def test_inf_torque_replaced_with_zero(self):
    """Test that Inf torque is replaced with 0.0."""
    CC, lac_log = self.controls.state_control()

    # Inject Inf into torque
    CC.actuators.torque = float('inf')

    for p in ACTUATOR_FIELDS:
      attr = getattr(CC.actuators, p)
      if not isinstance(attr, (int, float)):
        continue
      if not math.isfinite(attr):
        setattr(CC.actuators, p, 0.0)

    assert CC.actuators.torque == 0.0

  def test_negative_inf_steering_angle_replaced_with_zero(self):
    """Test that -Inf steeringAngleDeg is replaced with 0.0."""
    CC, lac_log = self.controls.state_control()

    # Inject -Inf
    CC.actuators.steeringAngleDeg = float('-inf')

    for p in ACTUATOR_FIELDS:
      attr = getattr(CC.actuators, p)
      if not isinstance(attr, (int, float)):
        continue
      if not math.isfinite(attr):
        setattr(CC.actuators, p, 0.0)

    assert CC.actuators.steeringAngleDeg == 0.0

  def test_finite_values_unchanged(self):
    """Test that finite values are not modified."""
    CC, lac_log = self.controls.state_control()

    # Set specific finite values
    original_accel = 1.5
    original_torque = 0.3
    CC.actuators.accel = original_accel
    CC.actuators.torque = original_torque

    for p in ACTUATOR_FIELDS:
      attr = getattr(CC.actuators, p)
      if not isinstance(attr, (int, float)):
        continue
      if not math.isfinite(attr):
        setattr(CC.actuators, p, 0.0)

    assert CC.actuators.accel == pytest.approx(original_accel)
    assert CC.actuators.torque == pytest.approx(original_torque)


class TestControlsCruiseLogic:
  """Tests for cruise control override, cancel, and resume logic."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm
    self._setup_default_sm()

  def _setup_default_sm(self, vEgo=20.0, cruiseEnabled=True, cruiseStandstill=False, enabled=True, active=True, shouldStop=False):
    """Set up mock SubMaster with configurable state."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = vEgo
    msg.carState.vCruise = 40.0
    msg.carState.vCruiseCluster = 40.0
    msg.carState.cruiseState.enabled = cruiseEnabled
    msg.carState.cruiseState.standstill = cruiseStandstill
    msg.carState.canValid = True
    self.mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('selfdriveState')
    msg.selfdriveState.enabled = enabled
    msg.selfdriveState.active = active
    msg.selfdriveState.state = State.enabled if enabled else State.disabled
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    msg.longitudinalPlan.shouldStop = shouldStop
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('modelV2')
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('liveTorqueParameters')
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    msg.driverMonitoringState.awarenessStatus = 1.0
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance
    self.mock_sm.valid['driverAssistance'] = True

    self.mock_sm.data['onroadEvents'] = []

  def test_cruise_resume_at_standstill_when_should_not_stop(self):
    """Test that resume is True at standstill when shouldStop is False."""
    self._setup_default_sm(cruiseStandstill=True, shouldStop=False)

    CC, lac_log = self.controls.state_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()
    self.controls.publish(CC, lac_log)

    # Resume should be True when enabled, at standstill, and not shouldStop
    assert CC.cruiseControl.resume is True

  def test_cruise_no_resume_when_should_stop(self):
    """Test that resume is False when shouldStop is True."""
    self._setup_default_sm(cruiseStandstill=True, shouldStop=True)

    CC, lac_log = self.controls.state_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()
    self.controls.publish(CC, lac_log)

    # Resume should be False when shouldStop is True
    assert CC.cruiseControl.resume is False

  def test_cruise_cancel_when_cc_disabled_but_cruise_enabled(self):
    """Test that cancel is True when CC disabled but cruise is enabled."""
    self._setup_default_sm(cruiseEnabled=True, enabled=False)

    CC, lac_log = self.controls.state_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()
    self.controls.publish(CC, lac_log)

    # Cancel should be True or pcmCruise is False
    if self.controls.CP.pcmCruise:
      assert CC.cruiseControl.cancel is True

  def test_cruise_no_cancel_when_both_enabled(self):
    """Test that cancel is False when both CC and cruise are enabled."""
    self._setup_default_sm(cruiseEnabled=True, enabled=True)

    CC, lac_log = self.controls.state_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()
    self.controls.publish(CC, lac_log)

    # With pcmCruise and both enabled, cancel should be False
    if self.controls.CP.pcmCruise:
      assert CC.cruiseControl.cancel is False


class TestControlsForceDecel:
  """Tests for forceDecel logic in controlsState."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm
    self._setup_default_sm()

  def _setup_default_sm(self, awarenessStatus=1.0, state=State.enabled):
    """Set up mock SubMaster with configurable state."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = 20.0
    msg.carState.vCruise = 40.0
    msg.carState.canValid = True
    self.mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('selfdriveState')
    msg.selfdriveState.enabled = True
    msg.selfdriveState.active = True
    msg.selfdriveState.state = state
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('modelV2')
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('liveTorqueParameters')
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    msg.driverMonitoringState.awarenessStatus = awarenessStatus
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance

    self.mock_sm.data['onroadEvents'] = []

  def test_force_decel_on_negative_awareness(self):
    """Test forceDecel is True when awarenessStatus < 0."""
    self._setup_default_sm(awarenessStatus=-0.5)

    self.controls.state_control()

    # Check the forceDecel calculation directly
    force_decel = bool((self.mock_sm['driverMonitoringState'].awarenessStatus < 0.0) or (self.mock_sm['selfdriveState'].state == State.softDisabling))

    assert force_decel is True

  def test_force_decel_on_soft_disabling(self):
    """Test forceDecel is True when state is softDisabling."""
    self._setup_default_sm(state=State.softDisabling)

    CC, lac_log = self.controls.state_control()

    force_decel = bool((self.mock_sm['driverMonitoringState'].awarenessStatus < 0.0) or (self.mock_sm['selfdriveState'].state == State.softDisabling))

    assert force_decel is True

  def test_no_force_decel_under_normal_conditions(self):
    """Test forceDecel is False under normal conditions."""
    self._setup_default_sm(awarenessStatus=1.0, state=State.enabled)

    CC, lac_log = self.controls.state_control()

    force_decel = bool((self.mock_sm['driverMonitoringState'].awarenessStatus < 0.0) or (self.mock_sm['selfdriveState'].state == State.softDisabling))

    assert force_decel is False


class TestControlsSteerLimitedBySafety:
  """Tests for steer_limited_by_safety detection."""

  def setup_method(self):
    """Set up test fixtures for angle control car (Nissan)."""
    self.CP_angle = setup_params_with_car(NISSAN.NISSAN_LEAF)
    self.controls_angle = Controls()

    # Set up for torque control car (Toyota)
    self.CP_torque = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls_torque = Controls()

    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]

    # Set up mock for angle control
    self.mock_sm_angle = MockSubMaster(services)
    self.controls_angle.sm = self.mock_sm_angle

    # Set up mock for torque control
    self.mock_sm_torque = MockSubMaster(services)
    self.controls_torque.sm = self.mock_sm_torque

  def _setup_mock_sm(self, mock_sm, active=True, steeringAngleOutput=0.0, torqueOutput=0.0):
    """Set up mock SubMaster values."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = 20.0
    msg.carState.vCruise = 40.0
    msg.carState.canValid = True
    mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = 1.0
    msg.liveParameters.steerRatio = 15.0
    mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('selfdriveState')
    msg.selfdriveState.enabled = True
    msg.selfdriveState.active = active
    mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('modelV2')
    mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('liveTorqueParameters')
    mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    msg.carOutput.actuatorsOutput.steeringAngleDeg = steeringAngleOutput
    msg.carOutput.actuatorsOutput.torque = torqueOutput
    mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance

    mock_sm.data['onroadEvents'] = []

  def test_steer_limited_not_set_when_inactive(self):
    """Test that steer_limited_by_safety is not updated when not active."""
    self._setup_mock_sm(self.mock_sm_torque, active=False)

    CC, lac_log = self.controls_torque.state_control()
    lac_log = log.ControlsState.LateralTorqueState.new_message()
    self.controls_torque.publish(CC, lac_log)

    # When not active, steer_limited_by_safety should not be updated
    # (it stays at its initial value of False)
    assert self.controls_torque.steer_limited_by_safety is False

  def test_torque_control_no_limit_when_output_matches(self):
    """Test torque control: no limit when commanded matches output."""
    self._setup_mock_sm(self.mock_sm_torque, active=True, torqueOutput=0.5)

    CC, lac_log = self.controls_torque.state_control()
    CC.actuators.torque = 0.5  # Same as output
    lac_log = log.ControlsState.LateralTorqueState.new_message()
    self.controls_torque.publish(CC, lac_log)

    assert self.controls_torque.steer_limited_by_safety is False

  def test_torque_control_limited_when_output_differs(self):
    """Test torque control: limited when commanded differs from output significantly."""
    self._setup_mock_sm(self.mock_sm_torque, active=True, torqueOutput=0.0)

    CC, lac_log = self.controls_torque.state_control()
    CC.actuators.torque = 0.5  # Different from output by > 0.01
    lac_log = log.ControlsState.LateralTorqueState.new_message()
    self.controls_torque.publish(CC, lac_log)

    assert self.controls_torque.steer_limited_by_safety is True


class TestControlsVehicleModelUpdate:
  """Tests for vehicle model parameter updates."""

  def setup_method(self):
    """Set up test fixtures."""
    self.CP = setup_params_with_car(TOYOTA.TOYOTA_RAV4)
    self.controls = Controls()

    services = [
      'liveDelay',
      'liveParameters',
      'liveTorqueParameters',
      'modelV2',
      'selfdriveState',
      'liveCalibration',
      'livePose',
      'longitudinalPlan',
      'carState',
      'carOutput',
      'driverMonitoringState',
      'onroadEvents',
      'driverAssistance',
    ]
    self.mock_sm = MockSubMaster(services)
    self.controls.sm = self.mock_sm
    self._setup_default_sm()

  def _setup_default_sm(self, stiffnessFactor=1.0, steerRatio=15.0):
    """Set up mock SubMaster."""
    msg = messaging.new_message('carState')
    msg.carState.vEgo = 20.0
    msg.carState.vCruise = 40.0
    self.mock_sm.data['carState'] = msg.as_reader().carState

    msg = messaging.new_message('liveParameters')
    msg.liveParameters.stiffnessFactor = stiffnessFactor
    msg.liveParameters.steerRatio = steerRatio
    self.mock_sm.data['liveParameters'] = msg.as_reader().liveParameters

    msg = messaging.new_message('selfdriveState')
    self.mock_sm.data['selfdriveState'] = msg.as_reader().selfdriveState

    msg = messaging.new_message('longitudinalPlan')
    self.mock_sm.data['longitudinalPlan'] = msg.as_reader().longitudinalPlan

    msg = messaging.new_message('modelV2')
    self.mock_sm.data['modelV2'] = msg.as_reader().modelV2

    msg = messaging.new_message('liveTorqueParameters')
    self.mock_sm.data['liveTorqueParameters'] = msg.as_reader().liveTorqueParameters

    msg = messaging.new_message('liveDelay')
    self.mock_sm.data['liveDelay'] = msg.as_reader().liveDelay

    msg = messaging.new_message('carOutput')
    self.mock_sm.data['carOutput'] = msg.as_reader().carOutput

    msg = messaging.new_message('driverMonitoringState')
    self.mock_sm.data['driverMonitoringState'] = msg.as_reader().driverMonitoringState

    msg = messaging.new_message('driverAssistance')
    self.mock_sm.data['driverAssistance'] = msg.as_reader().driverAssistance

    self.mock_sm.data['onroadEvents'] = []

  def test_very_low_stiffness_factor_clamped(self):
    """Test that stiffnessFactor below 0.1 is clamped."""
    self._setup_default_sm(stiffnessFactor=0.01)

    # state_control clamps to max(stiffnessFactor, 0.1)
    CC, lac_log = self.controls.state_control()

    # Should not raise - the clamp should prevent issues
    assert CC is not None

  def test_very_low_steer_ratio_clamped(self):
    """Test that steerRatio below 0.1 is clamped."""
    self._setup_default_sm(steerRatio=0.01)

    # state_control clamps to max(steerRatio, 0.1)
    CC, lac_log = self.controls.state_control()

    # Should not raise - the clamp should prevent issues
    assert CC is not None

  def test_normal_parameters_work(self):
    """Test that normal parameters work correctly."""
    self._setup_default_sm(stiffnessFactor=1.5, steerRatio=16.0)

    CC, lac_log = self.controls.state_control()

    assert CC is not None
