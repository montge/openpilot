"""Tests for selfdrive/modeld/parse_model_outputs.py."""
import numpy as np
import pytest

from openpilot.selfdrive.modeld.parse_model_outputs import (
  safe_exp, sigmoid, softmax, Parser,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


class TestSafeExp:
  """Test safe_exp function."""

  def test_normal_values(self):
    """Test safe_exp with normal values."""
    x = np.array([0.0, 1.0, 2.0])
    result = safe_exp(x)

    np.testing.assert_allclose(result, np.exp(x), rtol=1e-7)

  def test_clipping_large_values(self):
    """Test safe_exp clips large values to prevent overflow."""
    x = np.array([100.0, 1000.0])
    result = safe_exp(x)

    # Should be clipped to exp(11)
    expected = np.exp(11)
    np.testing.assert_allclose(result, [expected, expected], rtol=1e-7)

  def test_negative_values(self):
    """Test safe_exp with negative values."""
    x = np.array([-1.0, -5.0, -10.0])
    result = safe_exp(x)

    np.testing.assert_allclose(result, np.exp(x), rtol=1e-7)

  def test_out_parameter(self):
    """Test safe_exp with out parameter."""
    x = np.array([1.0, 2.0, 3.0])
    out = np.zeros_like(x)
    result = safe_exp(x, out=out)

    assert result is out
    np.testing.assert_allclose(out, np.exp(x), rtol=1e-7)

  def test_float16_friendly(self):
    """Test safe_exp with values that would overflow float16."""
    # exp(11) is about 60000, within float16 range
    x = np.array([10.0, 11.0, 12.0])
    result = safe_exp(x)

    # 12.0 should be clipped to 11.0
    assert result[2] == pytest.approx(np.exp(11), rel=1e-5)


class TestSigmoid:
  """Test sigmoid function."""

  def test_zero_input(self):
    """Test sigmoid(0) = 0.5."""
    result = sigmoid(np.array([0.0]))

    assert result[0] == 0.5

  def test_large_positive(self):
    """Test sigmoid approaches 1 for large positive values."""
    result = sigmoid(np.array([10.0]))

    assert result[0] == pytest.approx(1.0, abs=1e-4)

  def test_large_negative(self):
    """Test sigmoid approaches 0 for large negative values."""
    result = sigmoid(np.array([-10.0]))

    assert result[0] == pytest.approx(0.0, abs=1e-4)

  def test_symmetry(self):
    """Test sigmoid(-x) = 1 - sigmoid(x)."""
    x = np.array([2.0])
    assert sigmoid(-x)[0] == pytest.approx(1 - sigmoid(x)[0], rel=1e-6)

  def test_array_input(self):
    """Test sigmoid with array input."""
    x = np.array([-1.0, 0.0, 1.0])
    result = sigmoid(x)

    assert result.shape == (3,)
    assert result[0] < 0.5
    assert result[1] == 0.5
    assert result[2] > 0.5


class TestSoftmax:
  """Test softmax function."""

  def test_sums_to_one(self):
    """Test softmax output sums to 1."""
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)

    assert np.sum(result) == pytest.approx(1.0, rel=1e-6)

  def test_largest_gets_highest_prob(self):
    """Test largest input gets highest probability."""
    x = np.array([1.0, 5.0, 2.0])
    result = softmax(x)

    assert np.argmax(result) == 1

  def test_equal_inputs(self):
    """Test equal inputs give equal probabilities."""
    x = np.array([3.0, 3.0, 3.0])
    result = softmax(x)

    np.testing.assert_allclose(result, [1/3, 1/3, 1/3], rtol=1e-6)

  def test_2d_array(self):
    """Test softmax on 2D array along last axis."""
    x = np.array([[1.0, 2.0], [3.0, 1.0]])
    result = softmax(x)

    # Each row should sum to 1
    assert np.sum(result[0]) == pytest.approx(1.0, rel=1e-6)
    assert np.sum(result[1]) == pytest.approx(1.0, rel=1e-6)

  def test_numerical_stability(self):
    """Test softmax is numerically stable with large values."""
    x = np.array([1000.0, 1001.0, 1002.0])
    result = softmax(x)

    assert not np.isnan(result).any()
    assert np.sum(result) == pytest.approx(1.0, rel=1e-6)

  def test_axis_parameter(self):
    """Test softmax axis parameter."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = softmax(x, axis=0)

    # Each column should sum to 1
    np.testing.assert_allclose(np.sum(result, axis=0), [1.0, 1.0], rtol=1e-6)


class TestParser:
  """Test Parser class."""

  def test_init_default(self):
    """Test Parser default initialization."""
    parser = Parser()
    assert parser.ignore_missing is False

  def test_init_ignore_missing(self):
    """Test Parser with ignore_missing=True."""
    parser = Parser(ignore_missing=True)
    assert parser.ignore_missing is True

  def test_check_missing_raises(self):
    """Test check_missing raises for missing output."""
    parser = Parser(ignore_missing=False)
    outs = {}

    with pytest.raises(ValueError, match="Missing output"):
      parser.check_missing(outs, "test")

  def test_check_missing_ignores(self):
    """Test check_missing ignores when configured."""
    parser = Parser(ignore_missing=True)
    outs = {}

    result = parser.check_missing(outs, "test")
    assert result is True

  def test_check_missing_found(self):
    """Test check_missing returns False when output exists."""
    parser = Parser()
    outs = {"test": np.array([1, 2, 3])}

    result = parser.check_missing(outs, "test")
    assert result is False


class TestParserCategoricalCrossEntropy:
  """Test Parser.parse_categorical_crossentropy."""

  def test_applies_softmax(self):
    """Test categorical crossentropy applies softmax."""
    parser = Parser()
    outs = {"test": np.array([[1.0, 2.0, 3.0]])}

    parser.parse_categorical_crossentropy("test", outs)

    assert outs["test"].shape == (1, 3)
    assert np.sum(outs["test"]) == pytest.approx(1.0, rel=1e-6)

  def test_with_reshape(self):
    """Test categorical crossentropy with output reshape."""
    parser = Parser()
    outs = {"test": np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])}

    parser.parse_categorical_crossentropy("test", outs, out_shape=(2, 3))

    assert outs["test"].shape == (1, 2, 3)

  def test_missing_ignored(self):
    """Test missing output is ignored when configured."""
    parser = Parser(ignore_missing=True)
    outs = {}

    parser.parse_categorical_crossentropy("missing", outs)

    assert "missing" not in outs


class TestParserBinaryCrossEntropy:
  """Test Parser.parse_binary_crossentropy."""

  def test_applies_sigmoid(self):
    """Test binary crossentropy applies sigmoid."""
    parser = Parser()
    outs = {"test": np.array([[0.0]])}

    parser.parse_binary_crossentropy("test", outs)

    assert outs["test"][0, 0] == 0.5

  def test_range_zero_to_one(self):
    """Test output is in [0, 1] range."""
    parser = Parser()
    outs = {"test": np.array([[-10.0, 0.0, 10.0]])}

    parser.parse_binary_crossentropy("test", outs)

    assert np.all(outs["test"] >= 0)
    assert np.all(outs["test"] <= 1)


class TestParserMdn:
  """Test Parser.parse_mdn (Mixture Density Network)."""

  def test_basic_mdn(self):
    """Test basic MDN parsing with no hypotheses."""
    parser = Parser()
    # MDN output: n_values for mean, n_values for std
    # For out_shape=(3,), need 3 means + 3 stds = 6 values
    outs = {"test": np.array([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3]])}

    parser.parse_mdn("test", outs, in_N=0, out_N=0, out_shape=(3,))

    assert "test" in outs
    assert "test_stds" in outs

  def test_mdn_output_shape(self):
    """Test MDN output has correct shape."""
    parser = Parser()
    # For out_shape=(2,), need 2 means + 2 stds = 4 values
    outs = {"test": np.array([[1.0, 2.0, 0.5, 0.5]])}

    parser.parse_mdn("test", outs, in_N=0, out_N=0, out_shape=(2,))

    assert outs["test"].shape == (1, 2)
    assert outs["test_stds"].shape == (1, 2)


class TestParserIsMhp:
  """Test Parser.is_mhp (Multiple Hypothesis Prediction)."""

  def test_mhp_detection(self):
    """Test MHP detection based on shape."""
    parser = Parser()
    shape = 10

    # Non-MHP: shape[1] == 2 * shape
    outs_non_mhp = {"test": np.zeros((1, 2 * shape))}
    assert parser.is_mhp(outs_non_mhp, "test", shape) is False

    # MHP: shape[1] != 2 * shape
    outs_mhp = {"test": np.zeros((1, 3 * shape))}
    assert parser.is_mhp(outs_mhp, "test", shape) is True

  def test_is_mhp_missing(self):
    """Test is_mhp with missing output."""
    parser = Parser(ignore_missing=True)
    outs = {}

    result = parser.is_mhp(outs, "missing", 10)
    assert result is False


class TestModelConstants:
  """Test ModelConstants values."""

  def test_idx_n_value(self):
    """Test IDX_N is 33."""
    assert ModelConstants.IDX_N == 33

  def test_t_idxs_length(self):
    """Test T_IDXS has correct length."""
    assert len(ModelConstants.T_IDXS) == ModelConstants.IDX_N

  def test_x_idxs_length(self):
    """Test X_IDXS has correct length."""
    assert len(ModelConstants.X_IDXS) == ModelConstants.IDX_N

  def test_t_idxs_increasing(self):
    """Test T_IDXS is monotonically increasing."""
    for i in range(1, len(ModelConstants.T_IDXS)):
      assert ModelConstants.T_IDXS[i] >= ModelConstants.T_IDXS[i-1]

  def test_x_idxs_increasing(self):
    """Test X_IDXS is monotonically increasing."""
    for i in range(1, len(ModelConstants.X_IDXS)):
      assert ModelConstants.X_IDXS[i] >= ModelConstants.X_IDXS[i-1]

  def test_first_t_idx_is_zero(self):
    """Test first T_IDX is 0."""
    assert ModelConstants.T_IDXS[0] == 0.0

  def test_first_x_idx_is_zero(self):
    """Test first X_IDX is 0."""
    assert ModelConstants.X_IDXS[0] == 0.0

  def test_model_run_freq(self):
    """Test MODEL_RUN_FREQ is 20 Hz."""
    assert ModelConstants.MODEL_RUN_FREQ == 20

  def test_lead_t_idxs(self):
    """Test LEAD_T_IDXS values."""
    assert ModelConstants.LEAD_T_IDXS == [0., 2., 4., 6., 8., 10.]

  def test_fcw_thresholds(self):
    """Test FCW thresholds are arrays."""
    assert len(ModelConstants.FCW_THRESHOLDS_5MS2) == 5
    assert len(ModelConstants.FCW_THRESHOLDS_3MS2) == 2

  def test_feature_len(self):
    """Test FEATURE_LEN value."""
    assert ModelConstants.FEATURE_LEN == 512

  def test_desire_len(self):
    """Test DESIRE_LEN value."""
    assert ModelConstants.DESIRE_LEN == 8

  def test_num_lane_lines(self):
    """Test NUM_LANE_LINES is 4."""
    assert ModelConstants.NUM_LANE_LINES == 4

  def test_num_road_edges(self):
    """Test NUM_ROAD_EDGES is 2."""
    assert ModelConstants.NUM_ROAD_EDGES == 2
