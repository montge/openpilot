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
import sys
from pathlib import Path


def cmd_generate_scenarios(args):
  """Generate seed scenarios."""
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.scenario_generator import (
    save_seed_scenarios,
    generate_all_seed_scenarios,
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
  from openpilot.selfdrive.controls.lib.tests.algorithm_harness.metrics import format_metrics_table
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
      print(f"  {status} {result.scenario_name}: RMSE={result.metrics.tracking_error_rmse:.6f}, "
            f"Latency={result.metrics.latency_mean_ms:.2f}ms")
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

  return 0


def cmd_report(args):
  """Generate a report from results."""
  print("Report generation not yet implemented.")
  print("Use --format markdown or --format html")
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
  gen_parser.add_argument('--output', '-o', default='./scenarios',
                          help='Output directory for scenarios')
  gen_parser.set_defaults(func=cmd_generate_scenarios)

  # list
  list_parser = subparsers.add_parser('list', help='List available scenarios')
  list_parser.add_argument('--scenarios', '-s', default='./scenarios',
                           help='Directory containing scenarios')
  list_parser.set_defaults(func=cmd_list_scenarios)

  # run
  run_parser = subparsers.add_parser('run', help='Run algorithm against scenarios')
  run_parser.add_argument('--algorithm', '-a', required=True,
                          choices=['lateral_pid', 'lateral_torque', 'longitudinal'],
                          help='Algorithm to benchmark')
  run_parser.add_argument('--scenarios', '-s', default='./scenarios',
                          help='Directory containing scenarios')
  run_parser.set_defaults(func=cmd_run)

  # compare
  cmp_parser = subparsers.add_parser('compare', help='Compare two algorithms')
  cmp_parser.add_argument('--baseline', '-b', required=True,
                          choices=['lateral_pid', 'lateral_torque', 'longitudinal'],
                          help='Baseline algorithm')
  cmp_parser.add_argument('--candidate', '-c', required=True,
                          choices=['lateral_pid', 'lateral_torque', 'longitudinal'],
                          help='Candidate algorithm')
  cmp_parser.add_argument('--scenarios', '-s', default='./scenarios',
                          help='Directory containing scenarios')
  cmp_parser.set_defaults(func=cmd_compare)

  # report
  report_parser = subparsers.add_parser('report', help='Generate report from results')
  report_parser.add_argument('--results', '-r', required=True,
                             help='Results directory')
  report_parser.add_argument('--format', '-f', default='markdown',
                             choices=['markdown', 'html', 'json'],
                             help='Output format')
  report_parser.set_defaults(func=cmd_report)

  args = parser.parse_args()

  if args.command is None:
    parser.print_help()
    return 1

  return args.func(args)


if __name__ == '__main__':
  sys.exit(main())
