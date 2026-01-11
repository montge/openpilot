#!/usr/bin/env python3
"""
Algorithm Benchmarking CLI Tool

This tool provides command-line access to the algorithm test harness
for running benchmarks, comparing algorithms, and generating reports.

Usage:
  python tools/algo_bench.py generate-scenarios --output ./scenarios
  python tools/algo_bench.py run --algorithm lateral_pid --scenarios ./scenarios
  python tools/algo_bench.py compare --baseline lateral_pid --candidate lateral_torque --scenarios ./scenarios
  python tools/algo_bench.py report --results ./results --format markdown
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def cmd_generate_scenarios(args):
  """Generate seed scenarios."""
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_generator import (
    save_seed_scenarios,
  )

  output_dir = Path(args.output)
  print(f"Generating seed scenarios to: {output_dir}")

  try:
    saved_files = save_seed_scenarios(str(output_dir))
    print(f"\nGenerated {len(saved_files)} scenarios:")
    for f in saved_files:
      print(f"  - {f}")
  except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    print("Install with: pip install pandas pyarrow")
    return 1

  return 0


def cmd_list_scenarios(args):
  """List available scenarios."""
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import list_scenarios

  scenario_dir = Path(args.scenarios)
  if not scenario_dir.exists():
    print(f"Error: Scenario directory not found: {scenario_dir}")
    return 1

  scenarios = list_scenarios(scenario_dir)

  if not scenarios:
    print("No scenarios found.")
    return 0

  print(f"\nScenarios in {scenario_dir}:\n")
  print(f"{'Name':<30} {'Type':<20} {'Difficulty':<10} {'Duration':<10} {'Steps':<10}")
  print("-" * 80)

  for s in scenarios:
    if 'error' in s:
      print(f"{s['name']:<30} ERROR: {s['error']}")
    else:
      print(f"{s['name']:<30} {s['type']:<20} {s['difficulty']:<10} {s['duration_s']:<10.1f} {s['num_steps']:<10}")

  return 0


def cmd_run(args):
  """Run algorithm against scenarios."""
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import load_scenario, list_scenarios
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import ScenarioRunner
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.adapters import (
    LatControlPIDAdapter,
    LatControlTorqueAdapter,
    LongControlAdapter,
  )

  # Get algorithm
  algorithm_map = {
    'lateral_pid': LatControlPIDAdapter,
    'lateral_torque': LatControlTorqueAdapter,
    'longitudinal': LongControlAdapter,
  }

  if args.algorithm not in algorithm_map:
    print(f"Error: Unknown algorithm '{args.algorithm}'")
    print(f"Available: {', '.join(algorithm_map.keys())}")
    return 1

  algorithm_class = algorithm_map[args.algorithm]
  algorithm = algorithm_class()

  # Determine scenario class based on algorithm
  if args.algorithm.startswith('lateral'):
    scenario_class = 'lateral'
  else:
    scenario_class = 'longitudinal'

  # Load scenarios
  scenario_dir = Path(args.scenarios)
  if not scenario_dir.exists():
    print(f"Error: Scenario directory not found: {scenario_dir}")
    return 1

  scenario_infos = list_scenarios(scenario_dir)
  if not scenario_infos:
    print("No scenarios found.")
    return 1

  # Run benchmarks
  runner = ScenarioRunner(deterministic=True)
  results = []

  print(f"\nRunning {args.algorithm} against {len(scenario_infos)} scenarios...\n")

  for info in scenario_infos:
    if 'error' in info:
      print(f"  Skipping {info['name']}: {info['error']}")
      continue

    try:
      scenario = load_scenario(info['path'], scenario_class)
      result = runner.run(algorithm, scenario, args.algorithm)
      results.append(result)

      status = "✓" if result.success else "✗"
      print(f"  {status} {result.scenario_name}: RMSE={result.metrics.tracking_error_rmse:.6f}, Latency={result.metrics.latency_mean_ms:.2f}ms")
    except Exception as e:
      print(f"  ✗ {info['name']}: Error - {e}")

  # Print summary
  if results:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_steps = sum(r.metrics.total_steps for r in results)
    avg_rmse = sum(r.metrics.tracking_error_rmse for r in results) / len(results)
    avg_latency = sum(r.metrics.latency_mean_ms for r in results) / len(results)

    print(f"Scenarios run: {len(results)}")
    print(f"Total steps: {total_steps}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average latency: {avg_latency:.2f}ms")

    # Save results if output specified
    if args.output:
      output_dir = Path(args.output)
      output_file = save_results(
        results,
        output_dir,
        f"run_{args.algorithm}",
        metadata={'algorithm': args.algorithm, 'scenarios_dir': str(scenario_dir)},
      )
      print(f"\nResults saved to: {output_file}")

  return 0


def cmd_compare(args):
  """Compare two algorithms."""
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenarios import load_scenario, list_scenarios
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.runner import ScenarioRunner
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.adapters import (
    LatControlPIDAdapter,
    LatControlTorqueAdapter,
    LongControlAdapter,
  )

  algorithm_map = {
    'lateral_pid': LatControlPIDAdapter,
    'lateral_torque': LatControlTorqueAdapter,
    'longitudinal': LongControlAdapter,
  }

  if args.baseline not in algorithm_map:
    print(f"Error: Unknown baseline algorithm '{args.baseline}'")
    return 1

  if args.candidate not in algorithm_map:
    print(f"Error: Unknown candidate algorithm '{args.candidate}'")
    return 1

  baseline = algorithm_map[args.baseline]()
  candidate = algorithm_map[args.candidate]()

  # Load scenarios
  scenario_dir = Path(args.scenarios)
  scenario_infos = list_scenarios(scenario_dir)

  if not scenario_infos:
    print("No scenarios found.")
    return 1

  # Determine scenario class
  scenario_class = 'lateral' if args.baseline.startswith('lateral') else 'longitudinal'

  scenarios = []
  for info in scenario_infos:
    if 'error' not in info:
      try:
        scenarios.append(load_scenario(info['path'], scenario_class))
      except Exception:
        pass

  if not scenarios:
    print("No valid scenarios loaded.")
    return 1

  # Run comparison
  runner = ScenarioRunner(deterministic=True)

  print(f"\nComparing {args.baseline} vs {args.candidate} on {len(scenarios)} scenarios...\n")

  comparison = runner.compare(baseline, candidate, scenarios, args.baseline, args.candidate)

  # Print results
  print(f"{'Scenario':<30} {'Baseline RMSE':<15} {'Candidate RMSE':<15} {'Delta':<10} {'Better':<10}")
  print("-" * 80)

  for result in comparison['per_scenario']:
    base_rmse = result['baseline_metrics'].tracking_error_rmse
    cand_rmse = result['candidate_metrics'].tracking_error_rmse
    delta = cand_rmse - base_rmse
    better = "Candidate" if delta < 0 else "Baseline" if delta > 0 else "Tie"

    print(f"{result['scenario']:<30} {base_rmse:<15.6f} {cand_rmse:<15.6f} {delta:<+10.6f} {better:<10}")

  # Aggregate
  print("\n" + "=" * 60)
  print("AGGREGATE COMPARISON")
  print("=" * 60)

  agg = comparison['aggregate']
  for metric, data in agg['comparison'].items():
    if metric in ['tracking_error_rmse', 'latency_mean_ms', 'output_smoothness']:
      symbol = "↓" if data['improved'] else "↑"
      print(f"{metric}: {data['baseline']:.4f} → {data['candidate']:.4f} ({data['pct_change']:+.1f}%) {symbol}")

  # Save results if output specified
  if args.output:
    output_dir = Path(args.output)
    # Save both baseline and candidate results
    all_results = comparison['baseline_results'] + comparison['candidate_results']
    output_file = save_results(
      all_results,
      output_dir,
      f"compare_{args.baseline}_vs_{args.candidate}",
      metadata={
        'baseline': args.baseline,
        'candidate': args.candidate,
        'scenarios_dir': str(scenario_dir),
      },
    )
    print(f"\nResults saved to: {output_file}")

  return 0


def save_results(results: list, output_dir: Path, name: str, metadata: dict[str, Any] | None = None) -> Path:
  """Save benchmark results to JSON."""

  output_dir.mkdir(parents=True, exist_ok=True)

  # Convert results to serializable format
  serializable = []
  for r in results:
    result_dict = {
      'scenario_name': r.scenario_name,
      'algorithm_name': r.algorithm_name,
      'metrics': asdict(r.metrics),
      'outputs': r.outputs,
      'success': r.success,
      'error_message': r.error_message,
    }
    serializable.append(result_dict)

  data = {
    'name': name,
    'timestamp': datetime.now().isoformat(),
    'results': serializable,
    'metadata': metadata or {},
  }

  output_file = output_dir / f"{name}.json"
  with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

  return output_file


def load_results(results_path: Path) -> dict[str, Any]:
  """Load benchmark results from JSON."""
  with open(results_path) as f:
    return json.load(f)


def generate_markdown_report(data: dict[str, Any], include_plots: bool = False) -> str:
  """Generate a markdown report from results."""
  lines = [
    f"# Algorithm Benchmark Report: {data['name']}",
    "",
    f"**Generated:** {data['timestamp']}",
    "",
  ]

  if data.get('metadata'):
    meta = data['metadata']
    if 'algorithm' in meta:
      lines.append(f"**Algorithm:** {meta['algorithm']}")
    if 'baseline' in meta:
      lines.append(f"**Baseline:** {meta['baseline']}")
    if 'candidate' in meta:
      lines.append(f"**Candidate:** {meta['candidate']}")
    lines.append("")

  results = data.get('results', [])
  if not results:
    lines.append("*No results available.*")
    return "\n".join(lines)

  # Summary statistics
  successful = [r for r in results if r.get('success', True)]
  failed = [r for r in results if not r.get('success', True)]

  lines.extend(
    [
      "## Summary",
      "",
      f"- **Total scenarios:** {len(results)}",
      f"- **Successful:** {len(successful)}",
      f"- **Failed:** {len(failed)}",
      "",
    ]
  )

  if successful:
    avg_rmse = sum(r['metrics']['tracking_error_rmse'] for r in successful) / len(successful)
    avg_latency = sum(r['metrics']['latency_mean_ms'] for r in successful) / len(successful)
    total_steps = sum(r['metrics']['total_steps'] for r in successful)

    lines.extend(
      [
        "### Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Average RMSE | {avg_rmse:.6f} |",
        f"| Average Latency | {avg_latency:.3f} ms |",
        f"| Total Steps | {total_steps} |",
        "",
      ]
    )

  # Per-scenario results
  lines.extend(
    [
      "## Per-Scenario Results",
      "",
      "| Scenario | RMSE | Max Error | Latency (ms) | Smoothness | Status |",
      "|----------|------|-----------|--------------|------------|--------|",
    ]
  )

  for r in results:
    m = r['metrics']
    status = "✓" if r.get('success', True) else f"✗ {r.get('error_message', '')}"
    name, rmse, maxe = r['scenario_name'], m['tracking_error_rmse'], m['tracking_error_max']
    lat, smooth = m['latency_mean_ms'], m['output_smoothness']
    lines.append(f"| {name} | {rmse:.6f} | {maxe:.6f} | {lat:.3f} | {smooth:.6f} | {status} |")

  lines.append("")

  # Detailed metrics for each scenario
  lines.extend(
    [
      "## Detailed Metrics",
      "",
    ]
  )

  for r in results:
    m = r['metrics']
    lines.extend(
      [
        f"### {r['scenario_name']}",
        "",
        f"**Algorithm:** {r['algorithm_name']}",
        "",
        "#### Tracking Accuracy",
        f"- RMSE: {m['tracking_error_rmse']:.6f}",
        f"- Max Error: {m['tracking_error_max']:.6f}",
        f"- Mean Error: {m['tracking_error_mean']:.6f}",
        "",
        "#### Output Quality",
        f"- Smoothness (jerk RMS): {m['output_smoothness']:.6f}",
        f"- Std Dev: {m['output_std']:.6f}",
        "",
        "#### Latency",
        f"- Mean: {m['latency_mean_ms']:.3f} ms",
        f"- P50: {m['latency_p50_ms']:.3f} ms",
        f"- P99: {m['latency_p99_ms']:.3f} ms",
        f"- Max: {m['latency_max_ms']:.3f} ms",
        "",
        "#### Safety",
        f"- Saturation Ratio: {m['saturation_ratio']:.2%}",
        f"- Min Safety Margin: {m['safety_margin_min']:.6f}",
        "",
      ]
    )

  return "\n".join(lines)


def generate_html_report(data: dict[str, Any], include_plots: bool = False) -> str:
  """Generate an HTML report from results."""
  # Convert markdown to basic HTML
  md = generate_markdown_report(data, include_plots)

  html_lines = [
    "<!DOCTYPE html>",
    "<html>",
    "<head>",
    f"  <title>Benchmark Report: {data['name']}</title>",
    "  <style>",
    "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; ",
    "           max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }",
    "    h1 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }",
    "    h2 { color: #555; margin-top: 30px; }",
    "    h3 { color: #666; }",
    "    table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
    "    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }",
    "    th { background-color: #007acc; color: white; }",
    "    tr:nth-child(even) { background-color: #f9f9f9; }",
    "    tr:hover { background-color: #f1f1f1; }",
    "    code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }",
    "    .success { color: #28a745; }",
    "    .failure { color: #dc3545; }",
    "    .metric-card { background: #f8f9fa; border-radius: 8px; padding: 15px; margin: 10px 0; }",
    "  </style>",
    "</head>",
    "<body>",
  ]

  # Simple markdown to HTML conversion
  in_table = False
  for line in md.split('\n'):
    if line.startswith('# '):
      html_lines.append(f"  <h1>{line[2:]}</h1>")
    elif line.startswith('## '):
      html_lines.append(f"  <h2>{line[3:]}</h2>")
    elif line.startswith('### '):
      html_lines.append(f"  <h3>{line[4:]}</h3>")
    elif line.startswith('#### '):
      html_lines.append(f"  <h4>{line[5:]}</h4>")
    elif line.startswith('**') and line.endswith('**'):
      html_lines.append(f"  <p><strong>{line[2:-2]}</strong></p>")
    elif line.startswith('**') and '**' in line[2:]:
      # Bold text with trailing content
      parts = line.split('**')
      html_lines.append(f"  <p><strong>{parts[1]}</strong>{parts[2] if len(parts) > 2 else ''}</p>")
    elif line.startswith('- '):
      html_lines.append(f"  <li>{line[2:]}</li>")
    elif line.startswith('| ') and not in_table:
      in_table = True
      html_lines.append("  <table>")
      cells = [c.strip() for c in line.split('|')[1:-1]]
      html_lines.append("    <tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
    elif line.startswith(('|--', '| --')):
      pass  # Skip separator rows
    elif line.startswith('| ') and in_table:
      cells = [c.strip() for c in line.split('|')[1:-1]]
      # Color-code success/failure
      formatted_cells = []
      for c in cells:
        if c.startswith('✓'):
          formatted_cells.append(f'<span class="success">{c}</span>')
        elif c.startswith('✗'):
          formatted_cells.append(f'<span class="failure">{c}</span>')
        else:
          formatted_cells.append(c)
      html_lines.append("    <tr>" + "".join(f"<td>{c}</td>" for c in formatted_cells) + "</tr>")
    elif not line.startswith('|') and in_table:
      in_table = False
      html_lines.append("  </table>")
      if line.strip():
        html_lines.append(f"  <p>{line}</p>")
    elif line.startswith('*') and line.endswith('*'):
      html_lines.append(f"  <p><em>{line[1:-1]}</em></p>")
    elif line.strip():
      html_lines.append(f"  <p>{line}</p>")

  if in_table:
    html_lines.append("  </table>")

  html_lines.extend(
    [
      "</body>",
      "</html>",
    ]
  )

  return "\n".join(html_lines)


def create_plots(results: list[dict], output_dir: Path) -> list[Path]:
  """Create matplotlib visualizations for results."""
  try:
    import matplotlib

    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
  except ImportError:
    print("Warning: matplotlib not available, skipping plots")
    return []

  output_dir.mkdir(parents=True, exist_ok=True)
  plot_files = []

  # 1. RMSE comparison bar chart
  fig, ax = plt.subplots(figsize=(10, 6))
  scenarios = [r['scenario_name'] for r in results if r.get('success', True)]
  rmses = [r['metrics']['tracking_error_rmse'] for r in results if r.get('success', True)]

  if scenarios:
    bars = ax.bar(range(len(scenarios)), rmses, color='#007acc')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('RMSE')
    ax.set_title('Tracking Error (RMSE) by Scenario')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, rmses, strict=True):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    rmse_path = output_dir / 'rmse_comparison.png'
    plt.savefig(rmse_path, dpi=150)
    plt.close()
    plot_files.append(rmse_path)

  # 2. Latency distribution box plot
  fig, ax = plt.subplots(figsize=(10, 6))
  latency_data = []
  latency_labels = []

  for r in results:
    if r.get('success', True):
      m = r['metrics']
      # Approximate distribution from percentiles
      latency_data.append([m['latency_mean_ms'], m['latency_p50_ms'], m['latency_p99_ms'], m['latency_max_ms']])
      latency_labels.append(r['scenario_name'])

  if latency_data:
    positions = range(len(latency_labels))
    width = 0.2

    means = [d[0] for d in latency_data]
    p50s = [d[1] for d in latency_data]
    p99s = [d[2] for d in latency_data]
    maxs = [d[3] for d in latency_data]

    ax.bar([p - 1.5 * width for p in positions], means, width, label='Mean', color='#007acc')
    ax.bar([p - 0.5 * width for p in positions], p50s, width, label='P50', color='#28a745')
    ax.bar([p + 0.5 * width for p in positions], p99s, width, label='P99', color='#ffc107')
    ax.bar([p + 1.5 * width for p in positions], maxs, width, label='Max', color='#dc3545')

    ax.set_xticks(positions)
    ax.set_xticklabels(latency_labels, rotation=45, ha='right')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Distribution by Scenario')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    latency_path = output_dir / 'latency_distribution.png'
    plt.savefig(latency_path, dpi=150)
    plt.close()
    plot_files.append(latency_path)

  # 3. Metrics radar chart (if multiple scenarios)
  if len([r for r in results if r.get('success', True)]) >= 2:
    from math import pi

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = ['Accuracy\n(1/RMSE)', 'Smoothness\n(1/jerk)', 'Speed\n(steps/s)', 'Low Latency\n(1/ms)', 'Safety\n(margin)']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    for r in results[:5]:  # Limit to 5 scenarios for readability
      if not r.get('success', True):
        continue
      m = r['metrics']

      # Normalize metrics (higher is better)
      values = [
        1 / max(m['tracking_error_rmse'], 0.0001),
        1 / max(m['output_smoothness'], 0.0001),
        m['steps_per_second'] / 1000,  # Scale down
        1 / max(m['latency_mean_ms'], 0.01),
        max(m['safety_margin_min'], 0) if m['safety_margin_min'] != float('inf') else 1,
      ]

      # Normalize to 0-1 range for display
      max_vals = [max(v, 1) for v in values]
      values = [v / max(mv, 0.001) for v, mv in zip(values, max_vals, strict=True)]
      values += values[:1]

      ax.plot(angles, values, 'o-', linewidth=2, label=r['scenario_name'][:15])
      ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Performance Profile (normalized)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    radar_path = output_dir / 'performance_radar.png'
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append(radar_path)

  return plot_files


def cmd_report(args):
  """Generate a report from results."""
  results_path = Path(args.results)

  # Handle both file and directory inputs
  if results_path.is_dir():
    # Find all JSON result files in directory
    json_files = list(results_path.glob('*.json'))
    if not json_files:
      print(f"Error: No result files found in {results_path}")
      return 1
    # Use most recent
    results_path = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Using results file: {results_path}")

  if not results_path.exists():
    print(f"Error: Results file not found: {results_path}")
    return 1

  try:
    data = load_results(results_path)
  except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in results file: {e}")
    return 1

  # Generate plots if requested
  plot_files = []
  if args.plot:
    plot_dir = results_path.parent / 'plots'
    print(f"Generating plots to: {plot_dir}")
    plot_files = create_plots(data.get('results', []), plot_dir)
    if plot_files:
      print(f"Created {len(plot_files)} plots:")
      for pf in plot_files:
        print(f"  - {pf}")

  # Generate report
  if args.format == 'markdown':
    report = generate_markdown_report(data, include_plots=bool(plot_files))
    ext = 'md'
  elif args.format == 'html':
    report = generate_html_report(data, include_plots=bool(plot_files))
    ext = 'html'
  elif args.format == 'json':
    report = json.dumps(data, indent=2)
    ext = 'json'
  else:
    print(f"Error: Unknown format '{args.format}'")
    return 1

  # Output report
  if args.output:
    output_path = Path(args.output)
    if output_path.is_dir():
      output_path = output_path / f"report.{ext}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
      f.write(report)
    print(f"Report written to: {output_path}")
  else:
    print(report)

  return 0


def main():
  parser = argparse.ArgumentParser(
    description="Algorithm Benchmarking CLI Tool",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )

  subparsers = parser.add_subparsers(dest='command', help='Available commands')

  # generate-scenarios
  gen_parser = subparsers.add_parser('generate-scenarios', help='Generate seed scenarios')
  gen_parser.add_argument('--output', '-o', default='./scenarios', help='Output directory for scenarios')
  gen_parser.set_defaults(func=cmd_generate_scenarios)

  # list
  list_parser = subparsers.add_parser('list', help='List available scenarios')
  list_parser.add_argument('--scenarios', '-s', default='./scenarios', help='Directory containing scenarios')
  list_parser.set_defaults(func=cmd_list_scenarios)

  # run
  run_parser = subparsers.add_parser('run', help='Run algorithm against scenarios')
  run_parser.add_argument('--algorithm', '-a', required=True, choices=['lateral_pid', 'lateral_torque', 'longitudinal'], help='Algorithm to benchmark')
  run_parser.add_argument('--scenarios', '-s', default='./scenarios', help='Directory containing scenarios')
  run_parser.add_argument('--output', '-o', help='Directory to save results (JSON)')
  run_parser.set_defaults(func=cmd_run)

  # compare
  cmp_parser = subparsers.add_parser('compare', help='Compare two algorithms')
  cmp_parser.add_argument('--baseline', '-b', required=True, choices=['lateral_pid', 'lateral_torque', 'longitudinal'], help='Baseline algorithm')
  cmp_parser.add_argument('--candidate', '-c', required=True, choices=['lateral_pid', 'lateral_torque', 'longitudinal'], help='Candidate algorithm')
  cmp_parser.add_argument('--scenarios', '-s', default='./scenarios', help='Directory containing scenarios')
  cmp_parser.add_argument('--output', '-o', help='Directory to save results (JSON)')
  cmp_parser.set_defaults(func=cmd_compare)

  # report
  report_parser = subparsers.add_parser('report', help='Generate report from results')
  report_parser.add_argument('--results', '-r', required=True, help='Results file or directory')
  report_parser.add_argument('--format', '-f', default='markdown', choices=['markdown', 'html', 'json'], help='Output format')
  report_parser.add_argument('--output', '-o', help='Output file or directory for report')
  report_parser.add_argument('--plot', '-p', action='store_true', help='Generate matplotlib visualizations')
  report_parser.set_defaults(func=cmd_report)

  args = parser.parse_args()

  if args.command is None:
    parser.print_help()
    return 1

  return args.func(args)


if __name__ == '__main__':
  sys.exit(main())
