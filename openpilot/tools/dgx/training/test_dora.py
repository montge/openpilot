"""Unit tests for DoRA (Weight-Decomposed Low-Rank Adaptation) implementation.

Tests verify correctness of:
- DoRALinear layer initialization and forward pass
- DoRAConv2d layer initialization and forward pass
- Weight merging for inference
- Model adaptation utilities

Requires PyTorch - tests are skipped if torch is not installed.
"""

import pytest

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")
nn = torch.nn

from openpilot.tools.dgx.training.dora import (
  DoRAConv2d,
  DoRALinear,
  apply_dora_to_model,
  count_parameters,
  get_dora_parameters,
)


class TestDoRALinear:
  """Tests for DoRALinear layer."""

  def test_init_dimensions(self):
    """Test DoRALinear initializes with correct dimensions."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8, alpha=1.0)

    assert dora.in_features == 64
    assert dora.out_features == 128
    assert dora.rank == 8
    assert dora.base_weight.shape == (128, 64)
    assert dora.magnitude.shape == (128,)
    assert dora.lora_A.shape == (8, 64)
    assert dora.lora_B.shape == (128, 8)

  def test_init_with_bias(self):
    """Test DoRALinear with bias."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8, bias=True)
    assert dora.bias is not None
    assert dora.bias.shape == (128,)

  def test_init_without_bias(self):
    """Test DoRALinear without bias."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8, bias=False)
    assert dora.bias is None

  def test_from_linear(self):
    """Test creating DoRALinear from existing Linear layer."""
    linear = nn.Linear(64, 128)
    dora = DoRALinear.from_linear(linear, rank=8, alpha=2.0)

    assert dora.in_features == 64
    assert dora.out_features == 128
    assert dora.rank == 8
    assert dora.alpha == 2.0
    # Base weight should match original
    assert torch.allclose(dora.base_weight, linear.weight.data)
    # Magnitude should be initialized from weight norms
    expected_magnitude = linear.weight.data.norm(dim=1)
    assert torch.allclose(dora.magnitude, expected_magnitude)

  def test_forward_shape(self):
    """Test forward pass produces correct output shape."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8)
    x = torch.randn(32, 64)  # batch_size=32

    output = dora(x)

    assert output.shape == (32, 128)

  def test_forward_batched(self):
    """Test forward pass with multiple batch dimensions."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8)
    x = torch.randn(4, 8, 64)  # (batch, seq, features)

    output = dora(x)

    assert output.shape == (4, 8, 128)

  def test_forward_deterministic(self):
    """Test forward pass is deterministic (no dropout in eval mode)."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8, dropout=0.0)
    dora.eval()
    x = torch.randn(32, 64)

    output1 = dora(x)
    output2 = dora(x)

    assert torch.allclose(output1, output2)

  def test_merge_weights(self):
    """Test weight merging produces valid Linear layer."""
    linear = nn.Linear(64, 128)
    dora = DoRALinear.from_linear(linear, rank=8)

    # Modify LoRA weights slightly
    dora.lora_B.data.fill_(0.01)

    merged = dora.merge_weights()

    assert isinstance(merged, nn.Linear)
    assert merged.weight.shape == (128, 64)
    assert merged.bias.shape == (128,)

  def test_merge_weights_preserves_output(self):
    """Test merged weights produce same output as DoRA layer."""
    linear = nn.Linear(64, 128)
    dora = DoRALinear.from_linear(linear, rank=8)
    dora.eval()

    x = torch.randn(32, 64)

    dora_output = dora(x)
    merged = dora.merge_weights()
    merged_output = merged(x)

    assert torch.allclose(dora_output, merged_output, atol=1e-5)

  def test_trainable_parameters(self):
    """Test only DoRA parameters are trainable."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8)

    # Base weight should not require grad
    assert not dora.base_weight.requires_grad

    # DoRA parameters should require grad
    assert dora.magnitude.requires_grad
    assert dora.lora_A.requires_grad
    assert dora.lora_B.requires_grad


class TestDoRAConv2d:
  """Tests for DoRAConv2d layer."""

  def test_init_dimensions(self):
    """Test DoRAConv2d initializes with correct dimensions."""
    dora = DoRAConv2d(
      in_channels=32,
      out_channels=64,
      kernel_size=3,
      padding=1,
      rank=8,
    )

    assert dora.in_channels == 32
    assert dora.out_channels == 64
    assert dora.kernel_size == (3, 3)
    assert dora.base_weight.shape == (64, 32, 3, 3)
    assert dora.magnitude.shape == (64,)
    # weight_dim = 32 * 3 * 3 = 288
    assert dora.lora_A.shape == (8, 288)
    assert dora.lora_B.shape == (64, 8)

  def test_from_conv2d(self):
    """Test creating DoRAConv2d from existing Conv2d layer."""
    conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    dora = DoRAConv2d.from_conv2d(conv, rank=8)

    assert dora.in_channels == 32
    assert dora.out_channels == 64
    assert torch.allclose(dora.base_weight, conv.weight.data)

  def test_forward_shape(self):
    """Test forward pass produces correct output shape."""
    dora = DoRAConv2d(
      in_channels=32,
      out_channels=64,
      kernel_size=3,
      padding=1,
      rank=8,
    )
    x = torch.randn(4, 32, 16, 16)  # (batch, channels, h, w)

    output = dora(x)

    assert output.shape == (4, 64, 16, 16)

  def test_forward_stride(self):
    """Test forward pass with stride."""
    dora = DoRAConv2d(
      in_channels=32,
      out_channels=64,
      kernel_size=3,
      stride=2,
      padding=1,
      rank=8,
    )
    x = torch.randn(4, 32, 16, 16)

    output = dora(x)

    assert output.shape == (4, 64, 8, 8)


class TestApplyDoraToModel:
  """Tests for apply_dora_to_model utility."""

  def test_apply_to_linear_layers(self):
    """Test applying DoRA to all Linear layers."""
    model = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
    )

    adapted = apply_dora_to_model(model, rank=8)

    assert isinstance(adapted[0], DoRALinear)
    assert isinstance(adapted[1], nn.ReLU)
    assert isinstance(adapted[2], DoRALinear)

  def test_apply_to_conv_layers(self):
    """Test applying DoRA to Conv2d layers."""
    model = nn.Sequential(
      nn.Conv2d(3, 32, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 64, 3, padding=1),
    )

    adapted = apply_dora_to_model(model, rank=8)

    assert isinstance(adapted[0], DoRAConv2d)
    assert isinstance(adapted[2], DoRAConv2d)

  def test_apply_with_target_modules(self):
    """Test applying DoRA only to specific modules."""
    model = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
    )
    # Name the modules
    model[0] = nn.Linear(64, 128)
    model[2] = nn.Linear(128, 64)

    # Only adapt first layer (index 0)
    adapted = apply_dora_to_model(model, target_modules=["0"], rank=8)

    assert isinstance(adapted[0], DoRALinear)
    assert isinstance(adapted[2], nn.Linear)  # Not adapted

  def test_preserves_forward_functionality(self):
    """Test adapted model still produces valid outputs."""
    model = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
    )

    adapted = apply_dora_to_model(model, rank=8)
    x = torch.randn(4, 64)

    output = adapted(x)

    assert output.shape == (4, 32)
    assert not torch.isnan(output).any()


class TestParameterCounting:
  """Tests for parameter counting utilities."""

  def test_count_parameters(self):
    """Test parameter counting."""
    model = nn.Sequential(
      nn.Linear(64, 128),
      nn.Linear(128, 64),
    )

    adapted = apply_dora_to_model(model, rank=8)
    counts = count_parameters(adapted)

    assert "total" in counts
    assert "trainable" in counts
    assert "dora" in counts
    assert "frozen" in counts
    assert "dora_percent" in counts

    # DoRA params should be less than total
    assert counts["dora"] < counts["total"]
    # DoRA percent should be reasonable (< 100%)
    assert 0 < counts["dora_percent"] < 100

  def test_get_dora_parameters(self):
    """Test getting only DoRA parameters."""
    model = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
    )

    adapted = apply_dora_to_model(model, rank=8)
    dora_params = get_dora_parameters(adapted)

    # Should have magnitude, lora_A, lora_B, bias for each DoRALinear
    # 2 DoRALinear layers * 4 params each = 8 params
    assert len(dora_params) == 8

    # All should require grad
    for p in dora_params:
      assert p.requires_grad


class TestDoRAGradients:
  """Tests for gradient computation through DoRA layers."""

  def test_gradients_flow(self):
    """Test gradients flow through DoRA layer."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8)
    x = torch.randn(4, 64, requires_grad=True)

    output = dora(x)
    loss = output.sum()
    loss.backward()

    # Check gradients exist for trainable params
    assert dora.magnitude.grad is not None
    assert dora.lora_A.grad is not None
    assert dora.lora_B.grad is not None

  def test_base_weight_frozen(self):
    """Test base weight doesn't receive gradients."""
    linear = nn.Linear(64, 128)
    dora = DoRALinear.from_linear(linear, rank=8)
    x = torch.randn(4, 64)

    output = dora(x)
    loss = output.sum()
    loss.backward()

    # Base weight is a buffer, not a parameter
    assert not dora.base_weight.requires_grad


class TestDoRAScaling:
  """Tests for DoRA alpha/rank scaling."""

  def test_scaling_factor(self):
    """Test scaling factor is computed correctly."""
    dora = DoRALinear(in_features=64, out_features=128, rank=8, alpha=16.0)

    expected_scaling = 16.0 / 8  # alpha / rank = 2.0
    assert dora.scaling == expected_scaling

  def test_alpha_affects_output(self):
    """Test different alpha values affect output magnitude."""
    linear = nn.Linear(64, 128)

    dora_low = DoRALinear.from_linear(linear, rank=8, alpha=1.0)
    dora_high = DoRALinear.from_linear(linear, rank=8, alpha=16.0)

    # Set same LoRA weights
    with torch.no_grad():
      dora_low.lora_A.copy_(torch.randn_like(dora_low.lora_A))
      dora_low.lora_B.copy_(torch.randn_like(dora_low.lora_B))
      dora_high.lora_A.copy_(dora_low.lora_A)
      dora_high.lora_B.copy_(dora_low.lora_B)

    x = torch.randn(4, 64)

    out_low = dora_low(x)
    out_high = dora_high(x)

    # Outputs should differ due to different scaling
    assert not torch.allclose(out_low, out_high)
