#!/usr/bin/env python3
"""DoRA fine-tuning training script for openpilot models.

Usage:
  python tools/dgx/training/train.py --data /path/to/training/data
  python tools/dgx/training/train.py --data /path/to/data --epochs 10 --dora-rank 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.optim as optim  # type: ignore[import-not-found]
from torch.utils.data import DataLoader  # type: ignore[import-not-found]

# Local imports
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
  """Create dummy dataloader for testing the training loop.

  TODO: Replace with actual route log dataloader.
  """
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


def train_epoch(
  student: nn.Module,
  teacher,  # TeacherModel (TensorRT)
  dataloader: DataLoader,
  optimizer: optim.Optimizer,
  criterion: nn.Module,
  device: torch.device,
  epoch: int,
  log_interval: int = 10,
) -> dict[str, float]:
  """Train for one epoch."""
  student.train()

  total_loss = 0.0
  num_batches = 0
  start_time = time.perf_counter()

  for batch_idx, batch in enumerate(dataloader):
    # Move to device
    img = batch["img"].to(device)
    big_img = batch["big_img"].to(device)
    desire = batch["desire"].to(device)
    traffic = batch["traffic_convention"].to(device)

    # Generate teacher labels (no grad, uses TensorRT)
    with torch.no_grad():
      # Convert to numpy for TensorRT
      teacher_labels = teacher.generate_labels(
        img=img.cpu().numpy(),
        big_img=big_img.cpu().numpy(),
        desire=desire.cpu().numpy(),
        traffic_convention=traffic.cpu().numpy(),
      )

    # Student forward pass
    # TODO: Implement proper forward pass based on model architecture
    student_pred = student(img.float(), big_img.float())

    # Compute loss
    # TODO: Format predictions properly
    loss_dict = criterion(
      student_pred={"raw": student_pred},
      teacher_pred={"raw": torch.from_numpy(teacher_labels["raw_outputs"]).to(device)},
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
  parser.add_argument("--data", type=str, default=None, help="Training data directory")
  parser.add_argument("--model", type=str, default="selfdrive/modeld/models/driving_policy.onnx")
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

  # Setup teacher (TensorRT)
  print("\nLoading teacher model...")
  if args.dry_run:
    print("Dry run - using dummy teacher")
    teacher = None  # Would use dummy labels
  else:
    from openpilot.tools.dgx.training.teacher import create_teacher

    teacher = create_teacher(fp16=True)

  # Setup dataloader
  print("\nSetting up data...")
  if args.dry_run or args.data is None:
    print("Using dummy data for testing")
    dataloader = create_dummy_dataloader(args.batch_size, num_batches=50)
  else:
    # TODO: Implement actual route log dataloader
    raise NotImplementedError("Route log dataloader not yet implemented")

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
