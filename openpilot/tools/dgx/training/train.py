#!/usr/bin/env python3
"""DoRA fine-tuning training script for openpilot models.

Usage:
  python openpilot/tools/dgx/training/train.py --data /path/to/training/data
  python openpilot/tools/dgx/training/train.py --data /path/to/data --epochs 10 --dora-rank 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.optim as optim  # type: ignore[import-not-found]
from torch.utils.data import DataLoader  # type: ignore[import-not-found]

import numpy as np

# Local imports
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
from openpilot.tools.dgx.training.dora import apply_dora_to_model, count_parameters, get_dora_parameters
from openpilot.tools.dgx.training.losses import CombinedTrainingLoss


def load_student_model(onnx_path: str, device: torch.device) -> nn.Module:
  """Load student model from ONNX and convert to PyTorch.

  Uses onnx2pytorch for conversion, then wraps for training.
  """
  try:
    import onnx
    from onnx2pytorch import ConvertModel  # type: ignore[import-not-found]
  except ImportError:
    raise RuntimeError("Install onnx2pytorch: pip install onnx2pytorch") from None

  print(f"Loading student model from {onnx_path}...")
  onnx_model = onnx.load(onnx_path)
  pytorch_model = ConvertModel(onnx_model)
  pytorch_model = pytorch_model.to(device)

  return pytorch_model


def create_dummy_dataloader(batch_size: int, num_batches: int = 100):
  """Create dummy dataloader for testing the training loop."""
  from torch.utils.data import Dataset

  class DummyDataset(Dataset):
    def __init__(self, size: int):
      self.size = size

    def __len__(self):
      return self.size

    def __getitem__(self, idx):
      return {
        "img": torch.randint(0, 255, (12, 128, 256), dtype=torch.uint8),
        "big_img": torch.randint(0, 255, (12, 128, 256), dtype=torch.uint8),
        "desire": torch.randn(8, dtype=torch.float16),
        "traffic_convention": torch.randn(2, dtype=torch.float16),
      }

  dataset = DummyDataset(batch_size * num_batches)
  return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def create_route_dataloader(data_path: str, batch_size: int):
  """Create dataloader from route logs.

  Args:
    data_path: Path to segment data or comma car segments specification
    batch_size: Batch size for training

  Returns:
    PyTorch DataLoader
  """
  from openpilot.tools.dgx.training.dataloader import (
    CITestDataset,
    CommaCarSegmentsDataset,
    RouteLogDataset,
    create_dataloader,
  )

  # Check if using CI test segments
  dataset: RouteLogDataset
  if data_path == "ci":
    print("Using CI test segments for validation...")
    dataset = CITestDataset()
  # Check if using commaCarSegments
  elif data_path == "comma_car_segments":
    print("Using commaCarSegments dataset...")
    dataset = CommaCarSegmentsDataset(max_segments_per_platform=10)
  # Otherwise treat as local path
  else:
    # Parse as list of segment IDs from a file or directory
    data_path_obj = Path(data_path)
    if data_path_obj.is_file():
      # File containing segment IDs, one per line
      with open(data_path) as f:
        segments = [line.strip() for line in f if line.strip()]
    elif data_path_obj.is_dir():
      # Directory containing route data
      segments = [str(p) for p in data_path_obj.glob("*rlog*")]
    else:
      # Single segment ID
      segments = [data_path]

    dataset = RouteLogDataset(segments=segments)

  print(f"Dataset size: {len(dataset)} samples")
  return create_dataloader(dataset, batch_size=batch_size)


class DummyTeacher:
  """Zero-label stand-in for --dry-run, shaped like TeacherModel outputs."""

  def __init__(self, metadata: dict):
    self.input_shapes = metadata["input_shapes"]
    self.output_slices = metadata["output_slices"]
    self.output_len = next(iter(metadata["output_shapes"].values()))[-1]

  def generate_labels(self, img, big_img, desire, traffic_convention, **kwargs) -> dict[str, np.ndarray]:
    b = img.shape[0]
    return {
      "features": np.zeros((b, ModelConstants.FEATURE_LEN), dtype=np.float32),
      "raw_outputs": np.zeros((b, self.output_len), dtype=np.float32),
      "path_mean": np.zeros((b, 1, ModelConstants.IDX_N, 3), dtype=np.float32),
      "path_std": np.ones((b, 1, ModelConstants.IDX_N, 3), dtype=np.float32),
      "path_prob": np.ones((b, 1), dtype=np.float32),
    }


def student_forward(
  student: nn.Module,
  img: torch.Tensor,
  big_img: torch.Tensor,
  desire: torch.Tensor,
  traffic_convention: torch.Tensor,
  input_shapes: dict[str, tuple[int, ...]],
  device: torch.device,
) -> torch.Tensor:
  """Run the supercombo student and return flat (batch, N) outputs.

  The ONNX graph has fixed batch-1 shapes, so samples run one at a time.
  Inputs are passed positionally in graph-input order; desire goes into the
  last desire_pulse step (as modeld does), recurrent/action inputs are zero.
  """
  outs = []
  for i in range(img.shape[0]):
    inputs = []
    for name, shape in input_shapes.items():
      if name == "img":
        t = img[i : i + 1].float()
      elif name == "big_img":
        t = big_img[i : i + 1].float()
      elif name == "desire_pulse":
        t = torch.zeros(shape, dtype=torch.float32, device=device)
        t[0, -1, :] = desire[i].float()
      elif name == "traffic_convention":
        t = traffic_convention[i : i + 1].float()
      else:  # features_buffer, action_t: cold start
        t = torch.zeros(shape, dtype=torch.float32, device=device)
      inputs.append(t)
    out = student(*inputs)
    if isinstance(out, (list, tuple)):
      out = out[0]
    outs.append(out.reshape(1, -1))
  return torch.cat(outs, dim=0)


def extract_path_distribution(flat: torch.Tensor, output_slices: dict[str, slice]) -> tuple[torch.Tensor, torch.Tensor]:
  """Differentiably extract the plan position MDN from flat model outputs.

  Mirrors Parser.parse_mdn for the single-hypothesis plan head: first half of
  the slice is the mean, second half log-std. Returns path position mean/std
  shaped (batch, 1, IDX_N, 3) for the distillation loss.
  """
  plan_raw = flat[:, output_slices["plan"]]
  b = plan_raw.shape[0]
  n = plan_raw.shape[1] // 2
  shape = (b, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
  mean = plan_raw[:, :n].reshape(shape)
  std = torch.exp(plan_raw[:, n : 2 * n].clamp(max=11)).reshape(shape)
  return mean[:, None, :, Plan.POSITION], std[:, None, :, Plan.POSITION]


def train_epoch(
  student: nn.Module,
  teacher,  # TeacherModel (TensorRT) or DummyTeacher
  dataloader: DataLoader,
  optimizer: optim.Optimizer,
  criterion: nn.Module,
  device: torch.device,
  epoch: int,
  input_shapes: dict[str, tuple[int, ...]],
  output_slices: dict[str, slice],
  log_interval: int = 10,
) -> dict[str, float]:
  """Train for one epoch."""
  student.train()

  total_loss = 0.0
  num_batches = 0
  start_time = time.perf_counter()

  for batch_idx, batch in enumerate(dataloader):
    # Move to device (route dataloader emits road_frame/wide_frame)
    img = batch.get("img", batch.get("road_frame")).to(device)
    big_img = batch.get("big_img", batch.get("wide_frame")).to(device)
    desire = batch["desire"].to(device)
    traffic = batch["traffic_convention"].to(device)

    # The model wants two temporally stacked 6-channel frames (12 channels);
    # single-frame samples are duplicated as a cold-start approximation
    if img.shape[1] == 6:
      img = torch.cat([img, img], dim=1)
    if big_img.shape[1] == 6:
      big_img = torch.cat([big_img, big_img], dim=1)

    # Generate teacher labels (no grad, uses TensorRT)
    with torch.no_grad():
      # Convert to numpy for TensorRT
      teacher_labels = teacher.generate_labels(
        img=img.cpu().numpy(),
        big_img=big_img.cpu().numpy(),
        desire=desire.cpu().numpy(),
        traffic_convention=traffic.cpu().numpy(),
      )

    # Student forward pass: flat (batch, N) supercombo outputs
    student_flat = student_forward(student, img, big_img, desire, traffic, input_shapes, device)
    path_mean, path_std = extract_path_distribution(student_flat, output_slices)

    loss_dict = criterion(
      student_pred={"path_mean": path_mean, "path_std": path_std},
      teacher_pred={k: torch.from_numpy(teacher_labels[k]).float().to(device) for k in ("path_mean", "path_std", "path_prob")},
    )
    loss = loss_dict["total"]

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()

    total_loss += loss.item()
    num_batches += 1

    if batch_idx % log_interval == 0:
      elapsed = time.perf_counter() - start_time
      samples_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed
      print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f} Speed: {samples_per_sec:.1f} samples/sec")

  return {
    "loss": total_loss / num_batches,
    "time": time.perf_counter() - start_time,
  }


def main():
  parser = argparse.ArgumentParser(description="DoRA fine-tuning for openpilot")
  parser.add_argument("--data", type=str, default=None, help="Training data: path, 'ci' for CI segments, or 'comma_car_segments'")
  parser.add_argument("--model", type=str, default="openpilot/selfdrive/modeld/models/driving_supercombo.onnx")
  parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
  parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
  parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
  parser.add_argument("--dora-rank", type=int, default=16, help="DoRA rank")
  parser.add_argument("--dora-alpha", type=float, default=1.0, help="DoRA alpha")
  parser.add_argument("--output", type=str, default="checkpoints", help="Output directory")
  parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
  parser.add_argument("--dry-run", action="store_true", help="Test with dummy data")
  args = parser.parse_args()

  # Setup device
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Create output directory
  output_dir = Path(args.output)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Load student model
  student = load_student_model(args.model, device)

  # Apply DoRA adaptation
  print(f"\nApplying DoRA (rank={args.dora_rank}, alpha={args.dora_alpha})...")

  # Target specific layers for adaptation
  target_modules = ["Gemm", "fc", "proj", "out"]  # Adjust based on model architecture
  student = apply_dora_to_model(
    student,
    target_modules=target_modules,
    rank=args.dora_rank,
    alpha=args.dora_alpha,
  )

  # Print parameter counts
  param_counts = count_parameters(student)
  print(f"Total parameters: {param_counts['total']:,}")
  print(f"DoRA parameters: {param_counts['dora']:,} ({param_counts['dora_percent']:.2f}%)")
  print(f"Frozen parameters: {param_counts['frozen']:,}")

  # Setup optimizer (only DoRA parameters)
  dora_params = get_dora_parameters(student)
  optimizer = optim.AdamW(dora_params, lr=args.lr, weight_decay=1e-4)

  # Setup learning rate scheduler
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

  # Setup loss
  criterion = CombinedTrainingLoss(path_weight=1.0, feature_weight=0.1)

  # Model I/O specs from the ONNX-embedded metadata (input_shapes, output_slices)
  metadata = make_metadata_dict(args.model)

  # Setup teacher (TensorRT)
  print("\nLoading teacher model...")
  if args.dry_run:
    print("Dry run - using dummy teacher")
    teacher = DummyTeacher(metadata)
  else:
    from openpilot.tools.dgx.training.teacher import create_teacher

    teacher = create_teacher(fp16=True)

  # Setup dataloader
  print("\nSetting up data...")
  if args.dry_run or args.data is None:
    print("Using dummy data for testing")
    dataloader = create_dummy_dataloader(args.batch_size, num_batches=50)
  else:
    dataloader = create_route_dataloader(args.data, args.batch_size)

  # Training loop
  print(f"\nStarting training for {args.epochs} epochs...")
  print("=" * 60)

  best_loss = float("inf")

  for epoch in range(1, args.epochs + 1):
    # Train
    train_stats = train_epoch(
      student=student,
      teacher=teacher,
      dataloader=dataloader,
      optimizer=optimizer,
      criterion=criterion,
      device=device,
      epoch=epoch,
      input_shapes=metadata["input_shapes"],
      output_slices=metadata["output_slices"],
    )

    print(f"\nEpoch {epoch} complete:")
    print(f"  Loss: {train_stats['loss']:.4f}")
    print(f"  Time: {train_stats['time']:.1f}s")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Update scheduler
    scheduler.step(train_stats["loss"])

    # Save checkpoint
    if train_stats["loss"] < best_loss:
      best_loss = train_stats["loss"]
      checkpoint_path = output_dir / "best_model.pt"
      torch.save(
        {
          "epoch": epoch,
          "model_state_dict": student.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "loss": best_loss,
          "dora_rank": args.dora_rank,
          "dora_alpha": args.dora_alpha,
        },
        checkpoint_path,
      )
      print(f"  Saved best model to {checkpoint_path}")

    print("-" * 60)

  print("\nTraining complete!")
  print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
  main()
