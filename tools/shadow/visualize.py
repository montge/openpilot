#!/usr/bin/env python3
"""Visualization tools for shadow device comparison.

Creates time-series plots, heatmaps, and event timelines for comparing
shadow and production device outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from openpilot.tools.shadow.align import AlignedPair, AlignmentResult
  from openpilot.tools.shadow.metrics import ComparisonReport

# Check for matplotlib availability
try:
  import matplotlib
  matplotlib.use('Agg')  # Use non-interactive backend
  import matplotlib.pyplot as plt
  from matplotlib.figure import Figure
  import numpy as np
  MATPLOTLIB_AVAILABLE = True
except ImportError:
  MATPLOTLIB_AVAILABLE = False
  plt = None
  Figure = None


def check_matplotlib() -> None:
  """Check if matplotlib is available."""
  if not MATPLOTLIB_AVAILABLE:
    raise ImportError(
      "matplotlib is required for visualization. "
      "Install with: pip install matplotlib"
    )


def plot_time_series(
  pairs: list[AlignedPair],
  field: str,
  title: str | None = None,
  output_path: Path | None = None,
) -> Figure | None:
  """Plot time series comparison of shadow vs production.

  Args:
    pairs: List of aligned frame pairs
    field: Field to plot ("steer", "accel", "curvature")
    title: Optional plot title
    output_path: Path to save figure (if None, returns figure)

  Returns:
    matplotlib Figure or None if saved to file
  """
  check_matplotlib()
  from openpilot.tools.shadow.metrics import compute_time_series

  data = compute_time_series(pairs, field)

  fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

  # Plot values
  ax1 = axes[0]
  times = np.array(data["time"])
  # Normalize time to start at 0
  if len(times) > 0:
    times = times - times[0]

  ax1.plot(times, data["shadow"], label="Shadow", alpha=0.8, linewidth=0.8)
  ax1.plot(times, data["production"], label="Production", alpha=0.8, linewidth=0.8)
  ax1.set_ylabel(field.capitalize())
  ax1.legend(loc="upper right")
  ax1.grid(True, alpha=0.3)

  # Plot error
  ax2 = axes[1]
  ax2.fill_between(times, data["error"], alpha=0.5, label="Error")
  ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
  ax2.set_xlabel("Time (s)")
  ax2.set_ylabel("Error")
  ax2.legend(loc="upper right")
  ax2.grid(True, alpha=0.3)

  # Title
  if title:
    fig.suptitle(title, fontsize=14)
  else:
    fig.suptitle(f"{field.capitalize()} Comparison: Shadow vs Production", fontsize=14)

  plt.tight_layout()

  if output_path:
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None

  return fig


def plot_error_histogram(
  pairs: list[AlignedPair],
  field: str,
  output_path: Path | None = None,
) -> Figure | None:
  """Plot histogram of errors for a field.

  Args:
    pairs: List of aligned frame pairs
    field: Field to analyze
    output_path: Path to save figure

  Returns:
    matplotlib Figure or None if saved to file
  """
  check_matplotlib()
  from openpilot.tools.shadow.metrics import compute_time_series

  data = compute_time_series(pairs, field)
  errors = np.array(data["error"])

  fig, ax = plt.subplots(figsize=(10, 6))

  ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
  ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero')
  ax.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=1,
             label=f'Mean: {np.mean(errors):.4f}')

  ax.set_xlabel(f"{field.capitalize()} Error")
  ax.set_ylabel("Count")
  ax.set_title(f"{field.capitalize()} Error Distribution")
  ax.legend()
  ax.grid(True, alpha=0.3)

  # Add statistics text box
  stats_text = (
    f"N = {len(errors)}\n"
    f"Mean = {np.mean(errors):.4f}\n"
    f"Std = {np.std(errors):.4f}\n"
    f"Min = {np.min(errors):.4f}\n"
    f"Max = {np.max(errors):.4f}"
  )
  ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
          verticalalignment='top', horizontalalignment='right',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

  plt.tight_layout()

  if output_path:
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None

  return fig


def plot_control_heatmap(
  pairs: list[AlignedPair],
  output_path: Path | None = None,
) -> Figure | None:
  """Plot 2D heatmap of steer vs accel errors.

  Args:
    pairs: List of aligned frame pairs
    output_path: Path to save figure

  Returns:
    matplotlib Figure or None if saved to file
  """
  check_matplotlib()
  from openpilot.tools.shadow.metrics import compute_time_series

  steer_data = compute_time_series(pairs, "steer")
  accel_data = compute_time_series(pairs, "accel")

  steer_errors = np.array(steer_data["error"])
  accel_errors = np.array(accel_data["error"])

  fig, ax = plt.subplots(figsize=(10, 8))

  # Create 2D histogram
  h = ax.hist2d(steer_errors, accel_errors, bins=50, cmap='hot_r')
  plt.colorbar(h[3], ax=ax, label='Count')

  ax.axhline(y=0, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
  ax.axvline(x=0, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)

  ax.set_xlabel("Steer Error")
  ax.set_ylabel("Accel Error")
  ax.set_title("Control Error Heatmap")

  plt.tight_layout()

  if output_path:
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None

  return fig


def plot_correlation_scatter(
  pairs: list[AlignedPair],
  field: str,
  output_path: Path | None = None,
) -> Figure | None:
  """Plot scatter of shadow vs production values.

  Args:
    pairs: List of aligned frame pairs
    field: Field to analyze
    output_path: Path to save figure

  Returns:
    matplotlib Figure or None if saved to file
  """
  check_matplotlib()
  from openpilot.tools.shadow.metrics import compute_time_series

  data = compute_time_series(pairs, field)
  shadow = np.array(data["shadow"])
  production = np.array(data["production"])

  fig, ax = plt.subplots(figsize=(8, 8))

  # Scatter plot
  ax.scatter(production, shadow, alpha=0.3, s=1)

  # Perfect correlation line
  min_val = min(np.min(shadow), np.min(production))
  max_val = max(np.max(shadow), np.max(production))
  ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='Perfect match')

  # Compute correlation
  if len(shadow) > 1 and np.std(shadow) > 1e-10 and np.std(production) > 1e-10:
    corr = np.corrcoef(shadow, production)[0, 1]
    if np.isnan(corr):
      corr = 0.0
  else:
    corr = 0.0

  ax.set_xlabel(f"Production {field.capitalize()}")
  ax.set_ylabel(f"Shadow {field.capitalize()}")
  ax.set_title(f"{field.capitalize()} Correlation (r={corr:.4f})")
  ax.legend()
  ax.grid(True, alpha=0.3)
  ax.set_aspect('equal', adjustable='box')

  plt.tight_layout()

  if output_path:
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None

  return fig


def plot_event_timeline(
  pairs: list[AlignedPair],
  output_path: Path | None = None,
) -> Figure | None:
  """Plot event timeline comparison between shadow and production.

  Creates a horizontal timeline showing when events occurred on each device,
  allowing visual comparison of event timing and occurrence.

  Args:
    pairs: List of aligned frame pairs
    output_path: Path to save figure

  Returns:
    matplotlib Figure or None if saved to file
  """
  check_matplotlib()

  # Extract events with timestamps
  shadow_events: list[tuple[float, str]] = []
  prod_events: list[tuple[float, str]] = []

  # Get first timestamp for normalization
  if not pairs:
    return None

  first_time = pairs[0].shadow_frame.timestamp_mono

  for pair in pairs:
    time_offset = pair.shadow_frame.timestamp_mono - first_time

    # Shadow device events
    for event in pair.shadow_frame.events:
      shadow_events.append((time_offset, event))

    # Production device events
    for event in pair.production_frame.events:
      prod_events.append((time_offset, event))

  # If no events, create a simple message figure
  if not shadow_events and not prod_events:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.text(0.5, 0.5, "No events found in logs",
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title("Event Timeline Comparison")
    ax.axis('off')
    if output_path:
      fig.savefig(output_path, dpi=150, bbox_inches='tight')
      plt.close(fig)
      return None
    return fig

  # Get unique event types and assign colors
  all_event_types = sorted(set(e[1] for e in shadow_events + prod_events))
  color_map = plt.cm.get_cmap('tab20')
  event_colors = {event: color_map(i % 20) for i, event in enumerate(all_event_types)}

  fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

  # Plot shadow events
  ax_shadow = axes[0]
  ax_shadow.set_title("Shadow Device Events")
  ax_shadow.set_ylabel("Event Type")

  # Create y-positions for each event type
  event_y_map = {event: i for i, event in enumerate(all_event_types)}

  for time_s, event in shadow_events:
    y = event_y_map[event]
    ax_shadow.scatter(time_s, y, c=[event_colors[event]], s=50, alpha=0.7, marker='|')

  ax_shadow.set_yticks(range(len(all_event_types)))
  ax_shadow.set_yticklabels(all_event_types, fontsize=8)
  ax_shadow.grid(True, alpha=0.3, axis='x')

  # Plot production events
  ax_prod = axes[1]
  ax_prod.set_title("Production Device Events")
  ax_prod.set_xlabel("Time (s)")
  ax_prod.set_ylabel("Event Type")

  for time_s, event in prod_events:
    y = event_y_map[event]
    ax_prod.scatter(time_s, y, c=[event_colors[event]], s=50, alpha=0.7, marker='|')

  ax_prod.set_yticks(range(len(all_event_types)))
  ax_prod.set_yticklabels(all_event_types, fontsize=8)
  ax_prod.grid(True, alpha=0.3, axis='x')

  # Add summary counts
  shadow_count_by_type = {}
  prod_count_by_type = {}
  for _, event in shadow_events:
    shadow_count_by_type[event] = shadow_count_by_type.get(event, 0) + 1
  for _, event in prod_events:
    prod_count_by_type[event] = prod_count_by_type.get(event, 0) + 1

  summary = f"Shadow: {len(shadow_events)} events, Production: {len(prod_events)} events"
  fig.suptitle(f"Event Timeline Comparison\n{summary}", fontsize=12)

  plt.tight_layout()

  if output_path:
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None

  return fig


def plot_event_frequency(
  pairs: list[AlignedPair],
  output_path: Path | None = None,
) -> Figure | None:
  """Plot event frequency comparison as a bar chart.

  Args:
    pairs: List of aligned frame pairs
    output_path: Path to save figure

  Returns:
    matplotlib Figure or None if saved to file
  """
  check_matplotlib()

  # Count events by type
  shadow_counts: dict[str, int] = {}
  prod_counts: dict[str, int] = {}

  for pair in pairs:
    for event in pair.shadow_frame.events:
      shadow_counts[event] = shadow_counts.get(event, 0) + 1
    for event in pair.production_frame.events:
      prod_counts[event] = prod_counts.get(event, 0) + 1

  # Get all unique event types
  all_events = sorted(set(list(shadow_counts.keys()) + list(prod_counts.keys())))

  if not all_events:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, "No events found in logs",
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title("Event Frequency Comparison")
    ax.axis('off')
    if output_path:
      fig.savefig(output_path, dpi=150, bbox_inches='tight')
      plt.close(fig)
      return None
    return fig

  # Prepare data
  x = np.arange(len(all_events))
  width = 0.35

  shadow_values = [shadow_counts.get(e, 0) for e in all_events]
  prod_values = [prod_counts.get(e, 0) for e in all_events]

  fig, ax = plt.subplots(figsize=(max(10, len(all_events) * 0.8), 6))

  bars1 = ax.bar(x - width / 2, shadow_values, width, label='Shadow', alpha=0.8)
  bars2 = ax.bar(x + width / 2, prod_values, width, label='Production', alpha=0.8)

  ax.set_xlabel("Event Type")
  ax.set_ylabel("Count")
  ax.set_title("Event Frequency Comparison: Shadow vs Production")
  ax.set_xticks(x)
  ax.set_xticklabels(all_events, rotation=45, ha='right', fontsize=8)
  ax.legend()
  ax.grid(True, alpha=0.3, axis='y')

  # Add value labels on bars
  def add_labels(bars):
    for bar in bars:
      height = bar.get_height()
      if height > 0:
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

  add_labels(bars1)
  add_labels(bars2)

  plt.tight_layout()

  if output_path:
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None

  return fig


def plot_summary_dashboard(
  result: AlignmentResult,
  report: ComparisonReport,
  output_path: Path | None = None,
) -> Figure | None:
  """Create a summary dashboard with multiple plots.

  Args:
    result: AlignmentResult from log alignment
    report: ComparisonReport with metrics
    output_path: Path to save figure

  Returns:
    matplotlib Figure or None if saved to file
  """
  check_matplotlib()
  from openpilot.tools.shadow.metrics import compute_time_series

  fig = plt.figure(figsize=(16, 12))

  # Create grid for subplots
  gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

  # 1. Steer time series (top row, spans 2 columns)
  ax1 = fig.add_subplot(gs[0, :2])
  steer_data = compute_time_series(result.pairs, "steer")
  times = np.array(steer_data["time"])
  if len(times) > 0:
    times = times - times[0]
  ax1.plot(times, steer_data["shadow"], label="Shadow", alpha=0.8, linewidth=0.5)
  ax1.plot(times, steer_data["production"], label="Production", alpha=0.8, linewidth=0.5)
  ax1.set_ylabel("Steer")
  ax1.set_title("Steering Comparison")
  ax1.legend(loc="upper right")
  ax1.grid(True, alpha=0.3)

  # 2. Summary metrics (top right)
  ax2 = fig.add_subplot(gs[0, 2])
  ax2.axis('off')
  summary_text = (
    f"Alignment Summary\n"
    f"─────────────────\n"
    f"Aligned Pairs: {report.n_aligned_pairs}\n"
    f"Shadow Only: {report.n_shadow_only}\n"
    f"Production Only: {report.n_production_only}\n"
    f"Quality: {report.alignment_quality:.1%}\n"
    f"Method: {report.alignment_method}\n\n"
    f"Control Metrics\n"
    f"─────────────────\n"
    f"Steer RMSE: {report.control_metrics.steer_rmse:.4f}\n"
    f"Steer MAE: {report.control_metrics.steer_mae:.4f}\n"
    f"Accel RMSE: {report.control_metrics.accel_rmse:.4f}\n"
    f"Accel MAE: {report.control_metrics.accel_mae:.4f}"
  )
  ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

  # 3. Accel time series (middle row, spans 2 columns)
  ax3 = fig.add_subplot(gs[1, :2])
  accel_data = compute_time_series(result.pairs, "accel")
  ax3.plot(times, accel_data["shadow"], label="Shadow", alpha=0.8, linewidth=0.5)
  ax3.plot(times, accel_data["production"], label="Production", alpha=0.8, linewidth=0.5)
  ax3.set_ylabel("Accel (m/s²)")
  ax3.set_title("Acceleration Comparison")
  ax3.legend(loc="upper right")
  ax3.grid(True, alpha=0.3)

  # 4. Steer error histogram (middle right)
  ax4 = fig.add_subplot(gs[1, 2])
  ax4.hist(steer_data["error"], bins=30, edgecolor='black', alpha=0.7)
  ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
  ax4.set_xlabel("Steer Error")
  ax4.set_ylabel("Count")
  ax4.set_title("Steer Error Distribution")

  # 5. Error time series (bottom row, spans 2 columns)
  ax5 = fig.add_subplot(gs[2, :2])
  ax5.fill_between(times, steer_data["error"], alpha=0.5, label="Steer Error")
  ax5.fill_between(times, accel_data["error"], alpha=0.5, label="Accel Error")
  ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
  ax5.set_xlabel("Time (s)")
  ax5.set_ylabel("Error")
  ax5.set_title("Error Over Time")
  ax5.legend(loc="upper right")
  ax5.grid(True, alpha=0.3)

  # 6. Correlation scatter (bottom right)
  ax6 = fig.add_subplot(gs[2, 2])
  ax6.scatter(steer_data["production"], steer_data["shadow"], alpha=0.3, s=1)
  min_val = min(np.min(steer_data["shadow"]), np.min(steer_data["production"]))
  max_val = max(np.max(steer_data["shadow"]), np.max(steer_data["production"]))
  ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
  ax6.set_xlabel("Production Steer")
  ax6.set_ylabel("Shadow Steer")
  ax6.set_title("Steer Correlation")
  ax6.set_aspect('equal', adjustable='box')

  fig.suptitle("Shadow Device Comparison Dashboard", fontsize=16, y=0.98)

  if output_path:
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return None

  return fig


def generate_all_plots(
  result: AlignmentResult,
  report: ComparisonReport,
  output_dir: Path,
) -> list[Path]:
  """Generate all visualization plots and save to directory.

  Args:
    result: AlignmentResult from log alignment
    report: ComparisonReport with metrics
    output_dir: Directory to save plots

  Returns:
    List of generated file paths
  """
  check_matplotlib()

  output_dir.mkdir(parents=True, exist_ok=True)
  generated: list[Path] = []

  # Summary dashboard
  dashboard_path = output_dir / "dashboard.png"
  plot_summary_dashboard(result, report, dashboard_path)
  generated.append(dashboard_path)

  # Individual time series
  for field in ["steer", "accel", "curvature"]:
    ts_path = output_dir / f"timeseries_{field}.png"
    plot_time_series(result.pairs, field, output_path=ts_path)
    generated.append(ts_path)

    hist_path = output_dir / f"histogram_{field}.png"
    plot_error_histogram(result.pairs, field, output_path=hist_path)
    generated.append(hist_path)

    scatter_path = output_dir / f"correlation_{field}.png"
    plot_correlation_scatter(result.pairs, field, output_path=scatter_path)
    generated.append(scatter_path)

  # Control heatmap
  heatmap_path = output_dir / "control_heatmap.png"
  plot_control_heatmap(result.pairs, output_path=heatmap_path)
  generated.append(heatmap_path)

  # Event timeline
  timeline_path = output_dir / "event_timeline.png"
  plot_event_timeline(result.pairs, output_path=timeline_path)
  generated.append(timeline_path)

  # Event frequency
  frequency_path = output_dir / "event_frequency.png"
  plot_event_frequency(result.pairs, output_path=frequency_path)
  generated.append(frequency_path)

  return generated


def main():
  """CLI for visualization tools."""
  parser = argparse.ArgumentParser(
    description="Visualize shadow device comparison data",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Generate all plots from aligned logs:
  python visualize.py --shadow shadow.log --production prod.log --output ./plots

  # Generate single time series plot:
  python visualize.py --shadow shadow.log --production prod.log --plot timeseries --field steer
"""
  )
  parser.add_argument("--shadow", type=Path, help="Shadow device log file")
  parser.add_argument("--production", type=Path, help="Production device log file")
  parser.add_argument("--output", type=Path, default=Path("./plots"),
                      help="Output directory for plots")
  parser.add_argument("--plot", type=str, choices=["all", "timeseries", "histogram", "scatter", "heatmap", "dashboard", "events", "event_frequency"],
                      default="all", help="Type of plot to generate")
  parser.add_argument("--field", type=str, choices=["steer", "accel", "curvature"],
                      default="steer", help="Field to plot")
  parser.add_argument("--check", action="store_true",
                      help="Check if matplotlib is available")

  args = parser.parse_args()

  if args.check:
    if MATPLOTLIB_AVAILABLE:
      print("matplotlib is available")
      return 0
    else:
      print("matplotlib is NOT available")
      print("Install with: pip install matplotlib")
      return 1

  if not args.shadow or not args.production:
    print("Both --shadow and --production log files are required")
    parser.print_help()
    return 1

  check_matplotlib()

  # Load and align logs
  from openpilot.tools.shadow.align import align_logs
  from openpilot.tools.shadow.comparison_logger import ComparisonLogger
  from openpilot.tools.shadow.metrics import compute_all_metrics

  print(f"Loading shadow log: {args.shadow}")
  shadow_frames = ComparisonLogger.load_log(args.shadow)

  print(f"Loading production log: {args.production}")
  prod_frames = ComparisonLogger.load_log(args.production)

  print("Aligning logs...")
  result = align_logs(shadow_frames, prod_frames)
  report = compute_all_metrics(result)

  print(f"Aligned {len(result.pairs)} frame pairs")

  args.output.mkdir(parents=True, exist_ok=True)

  if args.plot == "all":
    generated = generate_all_plots(result, report, args.output)
    print(f"Generated {len(generated)} plots in {args.output}")
  elif args.plot == "dashboard":
    path = args.output / "dashboard.png"
    plot_summary_dashboard(result, report, path)
    print(f"Generated: {path}")
  elif args.plot == "timeseries":
    path = args.output / f"timeseries_{args.field}.png"
    plot_time_series(result.pairs, args.field, output_path=path)
    print(f"Generated: {path}")
  elif args.plot == "histogram":
    path = args.output / f"histogram_{args.field}.png"
    plot_error_histogram(result.pairs, args.field, output_path=path)
    print(f"Generated: {path}")
  elif args.plot == "scatter":
    path = args.output / f"correlation_{args.field}.png"
    plot_correlation_scatter(result.pairs, args.field, output_path=path)
    print(f"Generated: {path}")
  elif args.plot == "heatmap":
    path = args.output / "control_heatmap.png"
    plot_control_heatmap(result.pairs, output_path=path)
    print(f"Generated: {path}")
  elif args.plot == "events":
    path = args.output / "event_timeline.png"
    plot_event_timeline(result.pairs, output_path=path)
    print(f"Generated: {path}")
  elif args.plot == "event_frequency":
    path = args.output / "event_frequency.png"
    plot_event_frequency(result.pairs, output_path=path)
    print(f"Generated: {path}")

  return 0


if __name__ == "__main__":
  import sys
  sys.exit(main())
