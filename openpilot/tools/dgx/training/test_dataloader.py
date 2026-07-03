#!/usr/bin/env python3
"""Test script for route log dataloader.

Downloads a CI test segment and validates the dataloader works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

# Add openpilot to path if running standalone
if __name__ == "__main__":
  openpilot_root = Path(__file__).parent.parent.parent.parent
  sys.path.insert(0, str(openpilot_root))


# CI test segments for quick validation
CI_TEST_SEGMENTS = [
  "0982d79ebb0de295|2021-01-04--17-13-21--13",  # TOYOTA.TOYOTA_PRIUS
  "0982d79ebb0de295|2021-01-03--20-03-36--6",  # TOYOTA.TOYOTA_RAV4
  "eb140f119469d9ab|2021-06-12--10-46-24--27",  # HONDA.HONDA_CIVIC
]

CI_BASE_URL = "https://commadataci.blob.core.windows.net/openpilotci"


def download_ci_segment(segment_id: str, output_dir: Path) -> Path:
  """Download a CI test segment using openpilot's LogReader.

  This uses the existing infrastructure which handles URL construction and file access.
  """
  from openpilot.tools.lib.logreader import LogReader

  output_dir.mkdir(parents=True, exist_ok=True)
  output_path = output_dir / f"{segment_id.replace('|', '_').replace('--', '_')}_rlog.zst"

  if output_path.exists():
    print(f"Already downloaded: {output_path}")
    return output_path

  print(f"Using LogReader to access segment: {segment_id}")

  # LogReader will find and download the segment using openpilot's file sources
  lr = LogReader(segment_id)

  # Get the actual URL that was used
  if lr.logreader_identifiers:
    print(f"Found at: {lr.logreader_identifiers[0]}")

    # Download the file directly
    url = lr.logreader_identifiers[0]
    print(f"Downloading {url}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(output_path, "wb") as f:
      for chunk in resp.iter_content(chunk_size=8192):
        f.write(chunk)

    print(f"Downloaded to: {output_path}")
    return output_path
  else:
    raise RuntimeError(f"Could not find segment: {segment_id}")


def test_download_ci_segment():
  """Test downloading a CI segment."""
  output_dir = Path("/tmp/openpilot_test_segments")
  segment = CI_TEST_SEGMENTS[0]

  print(f"Downloading segment: {segment}")
  path = download_ci_segment(segment, output_dir)

  assert path.exists(), f"Downloaded file not found: {path}"
  assert path.stat().st_size > 0, f"Downloaded file is empty: {path}"

  print(f"Downloaded {path.stat().st_size / 1024 / 1024:.1f} MB to {path}")
  return path


def test_read_log():
  """Test reading a downloaded log file."""
  from openpilot.tools.lib.logreader import LogReader

  output_dir = Path("/tmp/openpilot_test_segments")
  log_files = list(output_dir.glob("*rlog*.zst"))

  if not log_files:
    print("No log files found. Run test_download_ci_segment first.")
    return

  log_path = log_files[0]
  print(f"Reading log: {log_path}")

  lr = LogReader(str(log_path))

  # Count message types
  msg_counts: dict[str, int] = {}
  for msg in lr:
    msg_type = msg.which()
    msg_counts[msg_type] = msg_counts.get(msg_type, 0) + 1

  print("\nMessage type counts:")
  for msg_type in sorted(msg_counts, key=lambda x: msg_counts[x], reverse=True)[:20]:
    print(f"  {msg_type}: {msg_counts[msg_type]}")

  # Check for key message types
  required = ["modelV2", "carState", "selfdriveState"]
  for req in required:
    if req in msg_counts:
      print(f"\n{req}: {msg_counts[req]} messages")
    else:
      print(f"\n{req}: NOT FOUND")


def test_extract_model_outputs():
  """Test extracting model outputs from logs."""
  from openpilot.tools.lib.logreader import LogReader

  output_dir = Path("/tmp/openpilot_test_segments")
  log_files = list(output_dir.glob("*rlog*.zst"))

  if not log_files:
    print("No log files found.")
    return

  log_path = log_files[0]
  print(f"Extracting model outputs from: {log_path}")

  lr = LogReader(str(log_path))

  model_outputs = []
  for msg in lr:
    if msg.which() == "modelV2":
      model = msg.modelV2
      output = {
        "timestamp": msg.logMonoTime,
        "position_x": list(model.position.x) if hasattr(model.position, "x") else [],
        "position_y": list(model.position.y) if hasattr(model.position, "y") else [],
        "velocity_x": list(model.velocity.x) if hasattr(model.velocity, "x") else [],
      }
      model_outputs.append(output)

      if len(model_outputs) >= 5:
        break

  print("\nFirst 5 model outputs:")
  for i, out in enumerate(model_outputs):
    print(f"  [{i}] timestamp: {out['timestamp']}")
    print(f"      position_x[:5]: {out['position_x'][:5]}")
    print(f"      position_y[:5]: {out['position_y'][:5]}")


def test_dataset_creation():
  """Test creating a dataset from CI segments."""
  from openpilot.tools.dgx.training.dataloader import CITestDataset

  print("Creating CITestDataset...")
  try:
    dataset = CITestDataset()
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) > 0:
      print("\nLoading first sample...")
      sample = dataset[0]
      print(f"  segment_id: {sample.segment_id}")
      print(f"  road_frame shape: {sample.road_frame.shape}")
      print(f"  wide_frame shape: {sample.wide_frame.shape}")
      print(f"  desire shape: {sample.desire.shape}")
      print(f"  traffic_convention: {sample.traffic_convention}")
      if sample.model_outputs is not None:
        print(f"  model_outputs shape: {sample.model_outputs.shape}")
  except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Test route log dataloader")
  parser.add_argument("--download", action="store_true", help="Download CI test segment")
  parser.add_argument("--read", action="store_true", help="Read downloaded log")
  parser.add_argument("--extract", action="store_true", help="Extract model outputs")
  parser.add_argument("--dataset", action="store_true", help="Test dataset creation")
  parser.add_argument("--all", action="store_true", help="Run all tests")
  args = parser.parse_args()

  if args.all or args.download:
    print("=" * 60)
    print("TEST: Download CI segment")
    print("=" * 60)
    test_download_ci_segment()
    print()

  if args.all or args.read:
    print("=" * 60)
    print("TEST: Read log file")
    print("=" * 60)
    test_read_log()
    print()

  if args.all or args.extract:
    print("=" * 60)
    print("TEST: Extract model outputs")
    print("=" * 60)
    test_extract_model_outputs()
    print()

  if args.all or args.dataset:
    print("=" * 60)
    print("TEST: Dataset creation")
    print("=" * 60)
    test_dataset_creation()
    print()

  if not any([args.download, args.read, args.extract, args.dataset, args.all]):
    print("Usage: python test_dataloader.py --all")
    print("       python test_dataloader.py --download  # Download CI segment")
    print("       python test_dataloader.py --read      # Read downloaded log")
    print("       python test_dataloader.py --extract   # Extract model outputs")
    print("       python test_dataloader.py --dataset   # Test dataset creation")
