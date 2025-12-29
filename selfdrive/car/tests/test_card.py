import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from cereal import car, log
from opendbc.car import structs
from opendbc.car.interfaces import CarInterfaceBase, RadarInterfaceBase
from openpilot.selfdrive.car.card import Car, obd_callback, can_comm_callbacks
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET, V_CRUISE_INITIAL

EventName = log.OnroadEvent.EventName


class MockCarInterface(CarInterfaceBase):
  """Mock car interface for testing"""

  def __init__(self, CP):
    self.CP = CP
    self.frame = 0
    self.CC = MagicMock()
    self.CS = MagicMock()

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, alpha_long, is_release, docs):
    return ret

  def update(self, can_list):
    """Return a mock CarState"""
    cs = structs.CarState()
    cs.vEgo = 10.0
    cs.canValid = True
    cs.cruiseState.available = True
    cs.cruiseState.speed = 30.0
    return cs

  def apply(self, CC, now_nanos):
    """Return mock actuators output and empty can sends"""
    return structs.CarControl.Actuators(), []

  def init(self, CP, can_recv, can_send):
    pass


class MockRadarInterface(RadarInterfaceBase):
  """Mock radar interface for testing"""

  def __init__(self, CP):
    self.CP = CP
    self.frame = 0

  def update(self, can_packets):
    self.frame += 1
    if (self.frame % 5) == 0:
      return structs.RadarData()
    return None


class TestOBDCallback:
  """Test OBD multiplexing callback function"""

  def test_obd_callback_sets_multiplexing_true(self):
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False  # Currently disabled

    callback = obd_callback(mock_params)
    callback(True)

    mock_params.put_bool.assert_called_with("ObdMultiplexingEnabled", True)
    mock_params.remove.assert_called_with("ObdMultiplexingChanged")

  def test_obd_callback_sets_multiplexing_false(self):
    mock_params = MagicMock()
    mock_params.get_bool.return_value = True  # Currently enabled

    callback = obd_callback(mock_params)
    callback(False)

    mock_params.put_bool.assert_called_with("ObdMultiplexingEnabled", False)

  def test_obd_callback_no_change_when_same(self):
    mock_params = MagicMock()
    mock_params.get_bool.return_value = True  # Already enabled

    callback = obd_callback(mock_params)
    callback(True)  # Request same state

    # Should not call put_bool since state is already correct
    mock_params.put_bool.assert_not_called()


class TestCanCommCallbacks:
  """Test CAN communication callbacks"""

  def test_can_recv_returns_empty_list_when_no_messages(self):
    mock_logcan = MagicMock()
    mock_sendcan = MagicMock()

    with patch('openpilot.selfdrive.car.card.messaging') as mock_messaging:
      mock_messaging.drain_sock.return_value = []

      can_recv, can_send = can_comm_callbacks(mock_logcan, mock_sendcan)
      result = can_recv(wait_for_one=False)

      assert result == []

  def test_can_recv_parses_can_messages(self):
    mock_logcan = MagicMock()
    mock_sendcan = MagicMock()

    # Create a mock CAN message
    mock_can_msg = MagicMock()
    mock_can_msg.address = 0x100
    mock_can_msg.dat = b'\x01\x02\x03'
    mock_can_msg.src = 0

    mock_can_frame = MagicMock()
    mock_can_frame.can = [mock_can_msg]

    with patch('openpilot.selfdrive.car.card.messaging') as mock_messaging:
      mock_messaging.drain_sock.return_value = [mock_can_frame]

      can_recv, can_send = can_comm_callbacks(mock_logcan, mock_sendcan)
      result = can_recv(wait_for_one=True)

      assert len(result) == 1
      assert len(result[0]) == 1
      assert result[0][0].address == 0x100
      assert result[0][0].dat == b'\x01\x02\x03'
      assert result[0][0].src == 0

  def test_can_send_sends_messages(self):
    mock_logcan = MagicMock()
    mock_sendcan = MagicMock()

    with patch('openpilot.selfdrive.car.card.messaging'):
      with patch('openpilot.selfdrive.car.card.can_list_to_can_capnp') as mock_to_capnp:
        mock_to_capnp.return_value = b'capnp_data'

        can_recv, can_send = can_comm_callbacks(mock_logcan, mock_sendcan)
        from opendbc.car.can_definitions import CanData
        can_send([CanData(0x100, b'\x01\x02', 0)])

        mock_sendcan.send.assert_called_once()


class TestCarInit:
  """Test Car class initialization with injected dependencies"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_car_init_with_injected_ci(self, mock_params_class, mock_messaging):
    """Test that Car initializes correctly with injected CarInterface"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    # Create mock CarParams
    CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      pcmCruise=True,
      openpilotLongitudinalControl=False,
    )

    # Create mock CarInterface
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP

    # Create mock RadarInterface
    mock_ri = MockRadarInterface(CP)

    # Initialize Car with injected interfaces
    car_obj = Car(CI=mock_ci, RI=mock_ri)

    assert car_obj.CI == mock_ci
    assert car_obj.RI == mock_ri
    assert car_obj.CP == CP
    assert car_obj.v_cruise_helper is not None
    assert car_obj.can_rcv_cum_timeout_counter == 0
    assert car_obj.initialized_prev is False

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_car_passive_mode_sets_no_output_safety(self, mock_params_class, mock_messaging):
    """Test that passive mode sets noOutput safety model"""
    mock_params = MagicMock()
    mock_params.get_bool.side_effect = lambda key: {
      "IsReleaseBranch": False,
      "OpenpilotEnabledToggle": False,  # Disables controller
      "IsMetric": True,
      "ExperimentalMode": False,
      "AlphaLongitudinalEnabled": False,
    }.get(key, False)
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      dashcamOnly=False,
    )

    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()  # Controller exists but toggle is off

    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # Should be passive since OpenpilotEnabledToggle is False
    assert car_obj.CP.passive is True
    assert len(car_obj.CP.safetyConfigs) == 1
    assert car_obj.CP.safetyConfigs[0].safetyModel == structs.CarParams.SafetyModel.noOutput

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_car_dashcam_only_is_passive(self, mock_params_class, mock_messaging):
    """Test that dashcamOnly cars are set to passive"""
    mock_params = MagicMock()
    mock_params.get_bool.side_effect = lambda key: {
      "IsReleaseBranch": False,
      "OpenpilotEnabledToggle": True,
      "IsMetric": True,
      "ExperimentalMode": False,
    }.get(key, False)
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      dashcamOnly=True,
    )

    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()

    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    assert car_obj.CP.passive is True


class TestCarStateUpdate:
  """Test Car state update functionality"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def setup_method(self, method, mock_params_class, mock_messaging):
    """Set up a Car instance for each test"""
    self.mock_params = MagicMock()
    self.mock_params.get_bool.side_effect = lambda key: {
      "IsReleaseBranch": False,
      "OpenpilotEnabledToggle": True,
      "IsMetric": True,
      "ExperimentalMode": False,
    }.get(key, False)
    self.mock_params.get.return_value = None
    mock_params_class.return_value = self.mock_params

    self.CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      pcmCruise=True,
    )

    self.mock_ci = MockCarInterface(self.CP)
    self.mock_ci.CP = self.CP
    self.mock_ci.CC = MagicMock()

    self.mock_ri = MockRadarInterface(self.CP)

    # We need to re-patch for each test since setup_method resets patches
    with patch('openpilot.selfdrive.car.card.messaging') as self.mock_messaging:
      with patch('openpilot.selfdrive.car.card.Params') as mock_params_class:
        mock_params_class.return_value = self.mock_params
        self.car = Car(CI=self.mock_ci, RI=self.mock_ri)

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_state_update_increments_timeout_counter_on_no_can(self, mock_params_class, mock_messaging):
    """Test that CAN timeout counter increments when no CAN messages received"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK", pcmCruise=True)
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # Mock no CAN messages
    mock_messaging.drain_sock_raw.return_value = []
    mock_messaging.SubMaster.return_value = MagicMock()

    with patch('openpilot.selfdrive.car.card.can_capnp_to_list', return_value=[]):
      initial_counter = car_obj.can_rcv_cum_timeout_counter
      car_obj.state_update()
      assert car_obj.can_rcv_cum_timeout_counter == initial_counter + 1


class TestVCruiseHelperIntegration:
  """Test VCruiseHelper integration with Car class"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_v_cruise_helper_initialization(self, mock_params_class, mock_messaging):
    """Test that VCruiseHelper is initialized with correct CarParams"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      pcmCruise=False,  # Non-PCM cruise for button control
    )

    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    assert car_obj.v_cruise_helper is not None
    assert car_obj.v_cruise_helper.CP.pcmCruise is False
    assert car_obj.v_cruise_helper.v_cruise_kph == V_CRUISE_UNSET

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_is_metric_and_experimental_mode_loaded(self, mock_params_class, mock_messaging):
    """Test that is_metric and experimental_mode are loaded from params"""
    mock_params = MagicMock()
    mock_params.get_bool.side_effect = lambda key: {
      "IsReleaseBranch": False,
      "OpenpilotEnabledToggle": True,
      "IsMetric": True,
      "ExperimentalMode": True,
    }.get(key, False)
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      openpilotLongitudinalControl=True,  # Required for experimental mode
    )

    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    assert car_obj.is_metric is True
    assert car_obj.experimental_mode is True


class TestControlsUpdate:
  """Test controls update functionality"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_controls_update_initializes_ci_on_first_run(self, mock_params_class, mock_messaging):
    """Test that CarInterface.init is called when controls first become ready"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK")
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ci.init = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)
    car_obj.initialized_prev = False

    # Mock sm.all_alive to return True
    car_obj.sm = MagicMock()
    car_obj.sm.all_alive.return_value = True

    # Create mock CarState and CarControl
    CS = car.CarState.new_message(canValid=True)
    CC = car.CarControl.new_message()

    with patch('openpilot.selfdrive.car.card.REPLAY', False):
      with patch('openpilot.selfdrive.car.card.time.monotonic', return_value=1000):
        car_obj.controls_update(CS, CC)

    # CI.init should have been called
    mock_ci.init.assert_called_once()
    # ControlsReady should be set
    mock_params.put_bool_nonblocking.assert_called_with("ControlsReady", True)


class TestStatePublish:
  """Test state publishing functionality"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_state_publish_sends_car_state(self, mock_params_class, mock_messaging):
    """Test that carState message is published"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK")
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # Mock the PubMaster
    car_obj.pm = MagicMock()
    car_obj.sm = MagicMock()
    car_obj.sm.all_checks.return_value = True
    car_obj.sm.frame = 1

    CS = car.CarState.new_message(canValid=True)

    car_obj.state_publish(CS, None)

    # Should send carState and carOutput
    call_args = [call[0][0] for call in car_obj.pm.send.call_args_list]
    assert 'carState' in call_args
    assert 'carOutput' in call_args

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_state_publish_sends_car_params_periodically(self, mock_params_class, mock_messaging):
    """Test that carParams is published every 50 seconds"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK")
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    car_obj.pm = MagicMock()
    car_obj.sm = MagicMock()
    car_obj.sm.all_checks.return_value = True
    # Frame 0 should trigger carParams publish (0 % 5000 == 0)
    car_obj.sm.frame = 0

    CS = car.CarState.new_message(canValid=True)
    car_obj.state_publish(CS, None)

    call_args = [call[0][0] for call in car_obj.pm.send.call_args_list]
    assert 'carParams' in call_args

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_state_publish_sends_live_tracks_when_radar_data(self, mock_params_class, mock_messaging):
    """Test that liveTracks is published when radar data is available"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK")
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    car_obj.pm = MagicMock()
    car_obj.sm = MagicMock()
    car_obj.sm.all_checks.return_value = True
    car_obj.sm.frame = 1

    CS = car.CarState.new_message(canValid=True)
    RD = structs.RadarData()

    car_obj.state_publish(CS, RD)

    call_args = [call[0][0] for call in car_obj.pm.send.call_args_list]
    assert 'liveTracks' in call_args


class TestSecOCKey:
  """Test SecOC key handling"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  @patch('builtins.open')
  def test_secoc_key_from_cache(self, mock_open, mock_params_class, mock_messaging):
    """Test that SecOC key is read from cache when available"""
    mock_params = MagicMock()
    mock_params.get_bool.side_effect = lambda key: {
      "IsReleaseBranch": False,  # Not release branch allows SecOC loading
      "OpenpilotEnabledToggle": True,
      "IsMetric": False,
      "ExperimentalMode": False,
    }.get(key, False)
    # Return a valid 32-character hex key (16 bytes)
    mock_params.get.side_effect = lambda key, **kwargs: {
      "SecOCKey": "0123456789ABCDEF0123456789ABCDEF",
    }.get(key)
    mock_params_class.return_value = mock_params

    # Mock file read for user key
    mock_file = MagicMock()
    mock_file.__enter__.return_value.readline.return_value = "0123456789ABCDEF0123456789ABCDEF"
    mock_open.return_value = mock_file

    CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      secOcRequired=True,
    )

    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ci.CS = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # SecOC key should be available
    assert car_obj.CP.secOcKeyAvailable is True


class TestCarParamsPersistence:
  """Test CarParams persistence functionality"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_car_params_cached(self, mock_params_class, mock_messaging):
    """Test that CarParams are cached"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK")
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # Should write CarParams to params
    put_calls = [call[0][0] for call in mock_params.put.call_args_list]
    assert "CarParams" in put_calls

    put_nonblocking_calls = [call[0][0] for call in mock_params.put_nonblocking.call_args_list]
    assert "CarParamsCache" in put_nonblocking_calls
    assert "CarParamsPersistent" in put_nonblocking_calls

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_prev_car_params_saved(self, mock_params_class, mock_messaging):
    """Test that previous route's CarParams are saved"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    # Return some previous CarParams data
    mock_params.get.side_effect = lambda key, **kwargs: {
      "CarParamsPersistent": b"prev_params_data",
    }.get(key)
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK")
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # Should save previous CarParams
    put_calls = [call[0] for call in mock_params.put.call_args_list]
    assert ("CarParamsPrevRoute", b"prev_params_data") in put_calls


class TestAlternativeExperience:
  """Test that alternativeExperience is reset"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_alternative_experience_reset(self, mock_params_class, mock_messaging):
    """Test that alternativeExperience is set to 0"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(
      carFingerprint="MOCK",
      alternativeExperience=5,  # Set some non-zero value
    )

    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # alternativeExperience should be reset to 0
    assert car_obj.CP.alternativeExperience == 0


class TestRatekeeper:
  """Test Ratekeeper configuration"""

  @patch('openpilot.selfdrive.car.card.messaging')
  @patch('openpilot.selfdrive.car.card.Params')
  def test_ratekeeper_100hz(self, mock_params_class, mock_messaging):
    """Test that Ratekeeper is configured for 100Hz"""
    mock_params = MagicMock()
    mock_params.get_bool.return_value = False
    mock_params.get.return_value = None
    mock_params_class.return_value = mock_params

    CP = car.CarParams.new_message(carFingerprint="MOCK")
    mock_ci = MockCarInterface(CP)
    mock_ci.CP = CP
    mock_ci.CC = MagicMock()
    mock_ri = MockRadarInterface(CP)

    car_obj = Car(CI=mock_ci, RI=mock_ri)

    # Ratekeeper should be configured for 100Hz (interval = 0.01s)
    assert car_obj.rk._interval == pytest.approx(1.0 / 100, rel=1e-6)
