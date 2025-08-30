#!/usr/bin/env python3
"""
ISRU Strategy Simulation System Main Entry
Refactored strategy simulation and comparison analysis system

Quick Start Examples:
    # Run 50-year time horizon strategy comparison simulation with visualization
    python strategies/main.py --time-horizon 50 --visualize --n-simulations 100

    # Run 10-year time horizon strategy comparison simulation (default)
    python strategies/main.py --visualize

    # Run single strategy Monte Carlo simulation
    python strategies/main.py monte-carlo --strategy flexible_deployment --time-horizon 30 --n-simulations 500 --visualize

    # Run strategy comparison analysis
    python strategies/main.py compare --time-horizon 25 --n-simulations 200 --visualize --save

    # Display visualization charts for existing results
    python strategies/main.py visualize

    # Export results to Excel
    python strategies/main.py results export --strategies upfront_deployment gradual_deployment flexible_deployment --time-horizons 10 20 30

Main Features:
    - Strategy Simulation: Support for Upfront Deployment, Gradual Deployment, Flexible Deployment ISRU strategies
    - Time Horizon Analysis: Configurable 10-50 year simulation time horizons
    - Monte Carlo Simulation: Support for multiple random simulations to evaluate strategy robustness
    - Visualization Analysis: Automatically generate decision variables, cost analysis charts
    - Result Management: Support for result saving, loading, exporting functions
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.simulation_engine import StrategySimulationEngine
from strategies.analysis.batch_runner import BatchSimulationRunner
from strategies.analysis.performance_analyzer import PerformanceAnalyzer
from strategies.utils.terminal_display import TerminalDisplay
from strategies.utils.result_manager import ResultManager

# Import visualization module
try:
    from strategies.visualization.strategy_visualizer import DecisionVariablesPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization module unavailable: {e}")
    VISUALIZATION_AVAILABLE = False


def load_parameters() -> dict:
    """Load system parameters"""
    params_path = project_root / "data" / "parameters.json"
    with open(params_path, 'r') as f:
        return json.load(f)


def show_visualization(results_dir: str = "strategies/simulation_results", time_horizon: int = 10):
    """
    Display visualization charts
    
    Args:
        results_dir: Results directory path
        time_horizon: Time horizon (years)
    """
    if not VISUALIZATION_AVAILABLE:
        print("ERROR: Visualization feature unavailable, please check if matplotlib and other dependencies are installed")
        return
    
    try:
        TerminalDisplay.print_section("Generating Visualization Charts")
        
        # Create visualizer
        plotter = DecisionVariablesPlotter(figsize=(16, 12))
        
        # Generate comprehensive dashboard, pass time horizon parameter
        figures = plotter.create_comprehensive_dashboard(results_dir, time_horizon)
        
        if figures:
            print(f"SUCCESS: Successfully generated {len(figures)} charts")
            print("INFO: Charts displayed, close chart windows to continue...")
            
            # Wait for user to close charts
            try:
                import matplotlib.pyplot as plt
                # Keep charts displayed until user closes them
                plt.show(block=True)
            except KeyboardInterrupt:
                print("\nUser interrupted, closing all charts...")
                plt.close('all')
        else:
            print("ERROR: Failed to generate charts, please check if simulation data is available")
            
    except Exception as e:
        print(f"ERROR: Error occurred during visualization: {e}")
        print("Please ensure simulation has been run and result data has been generated")


def run_default_simulation_with_visualization(args):
    """
    Run default T=10 strategy comparison simulation with visualization
    
    Args:
        args: Command line arguments
    """
    TerminalDisplay.print_header("ISRU Strategy Simulation System - Default Mode", width=70)
    
    # Set default parameters
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    # Run strategy comparison (T=10, three strategies)
    strategies = ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    time_horizon = args.time_horizon
    n_simulations = getattr(args, 'n_simulations', 50)  # Default 50 simulations
    
    TerminalDisplay.print_section(f"Running T={time_horizon} Strategy Comparison Simulation")
    print(f"Strategies: {', '.join([s.title() for s in strategies])}")
    print(f"Simulations: {n_simulations}")
    print(f"Time Horizon: {time_horizon} years")
    
    try:
        # Run strategy comparison
        comparison_results = runner.run_strategy_comparison(
            strategies=strategies,
            T=time_horizon,
            n_simulations=n_simulations,
            base_seed=getattr(args, 'seed', 42),
            save_results=True,  # Force save results for visualization
            show_progress=True
        )
        
        # Display brief results
        TerminalDisplay.print_section("Simulation Complete - Results Summary")
        for strategy_name, (results, stats) in comparison_results.items():
            summary_data = {
                "Strategy": strategy_name.title(),
                "NPV Mean": f"{stats['npv_mean']/70000:.1f}0K GBP",
                "Success Rate": f"{stats['probability_positive_npv']:.1%}",
                "Average Utilization": f"{stats['utilization_mean']:.1%}"
            }
            TerminalDisplay.print_summary_box(f"{strategy_name.title()} Strategy", summary_data)
        
        # Display visualization
        if args.visualize:
            show_visualization(time_horizon=time_horizon)
        else:
            print("\nTIP: Use --visualize parameter to view chart analysis")
            
    except Exception as e:
        print(f"ERROR: Error occurred during simulation: {e}")
        return


def run_single_simulation(args):
    """Run single simulation"""
    TerminalDisplay.print_header("Single Strategy Simulation", width=70)
    
    params = load_parameters()
    engine = StrategySimulationEngine(params)
    
    result = engine.run_single_simulation(
        strategy_name=args.strategy,
        T=args.time_horizon,
        seed=args.seed
    )
    
    # Display results
    TerminalDisplay.print_section(f"{args.strategy.title()} Strategy Simulation Results")
    
    summary_data = {
        "Strategy": args.strategy.title(),
        "Time Horizon": f"{args.time_horizon} years",
        "NPV": result.performance_metrics.get('npv', 0),
        "Average Utilization": result.performance_metrics.get('avg_utilization', 0),
        "Self-Sufficiency Rate": result.performance_metrics.get('self_sufficiency_rate', 0),
        "Total Cost": result.performance_metrics.get('total_cost', 0)
    }
    
    TerminalDisplay.print_summary_box("Simulation Results", summary_data)
    
    # Save results
    if args.save:
        output_dir = Path("strategies/simulation_results") / f"T{args.time_horizon}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{args.strategy}_single_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")


def run_monte_carlo_simulation(args):
    """Run Monte Carlo simulation"""
    TerminalDisplay.print_header("Monte Carlo Strategy Simulation", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    results, stats = runner.run_single_batch(
        strategy_name=args.strategy,
        T=args.time_horizon,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=args.save,
        show_progress=True
    )
    
    # Display statistical results
    TerminalDisplay.print_section("Monte Carlo Simulation Statistical Results")
    
    stats_data = {
        "Simulations": len(results),
        "NPV Mean": stats['npv_mean'],
        "NPV Std Dev": stats['npv_std'],
        "Success Rate": stats['probability_positive_npv'],
        "Average Utilization": stats['utilization_mean']
    }
    
    TerminalDisplay.print_summary_box(f"{args.strategy.title()} Strategy Statistics", stats_data)


def run_strategy_comparison(args):
    """Run strategy comparison"""
    TerminalDisplay.print_header("Strategy Comparison Analysis", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    strategies = args.strategies if args.strategies else ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    
    comparison_results = runner.run_strategy_comparison(
        strategies=strategies,
        T=args.time_horizon,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=args.save,
        show_progress=True
    )
    
    # Performance analysis
    if args.detailed_analysis:
        TerminalDisplay.print_section("Detailed Performance Analysis")
        
        analyzer = PerformanceAnalyzer()
        strategy_results = {name: results for name, (results, _) in comparison_results.items()}
        
        # Generate analysis report
        if args.save:
            report_file = Path("strategies/simulation_results") / f"comparison_report_T{args.time_horizon}.txt"
            report = analyzer.generate_performance_report(strategy_results, report_file)
            print(f"Detailed report saved to: {report_file}")
        else:
            report = analyzer.generate_performance_report(strategy_results)
            print(report)


def run_time_horizon_analysis(args):
    """Run time horizon analysis"""
    TerminalDisplay.print_header("Time Horizon Impact Analysis", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    time_horizons = args.time_horizons if args.time_horizons else [10, 20, 30, 40, 50]
    strategies = args.strategies if args.strategies else ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    
    horizon_results = runner.run_time_horizon_analysis(
        time_horizons=time_horizons,
        strategies=strategies,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=args.save,
        show_progress=True
    )
    
    # Time horizon impact analysis
    if args.detailed_analysis:
        TerminalDisplay.print_section("Time Horizon Impact Analysis")
        
        analyzer = PerformanceAnalyzer()
        
        # Reorganize data format
        formatted_results = {}
        for T, strategy_data in horizon_results.items():
            formatted_results[T] = {name: results for name, (results, _) in strategy_data.items()}
        
        horizon_analysis = analyzer.analyze_time_horizon_impact(formatted_results)
        
        # Display trend analysis
        trends = horizon_analysis.get('trends', {})
        for strategy, trend_data in trends.items():
            trend_direction = "increasing" if trend_data['is_improving'] else "decreasing"
            print(f"{strategy.title()} Strategy: NPV shows {trend_direction} trend with time horizon")


def run_parallel_batch(args):
    """Run parallel batch simulation"""
    TerminalDisplay.print_header("Parallel Batch Simulation", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    time_horizons = args.time_horizons if args.time_horizons else [10, 20, 30, 40, 50]
    strategies = args.strategies if args.strategies else ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    
    results = runner.run_parallel_batch(
        strategies=strategies,
        time_horizons=time_horizons,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        max_workers=args.max_workers,
        save_results=args.save
    )
    
    TerminalDisplay.print_section("Parallel Simulation Complete")
    print(f"Completed {len(strategies)} strategies × {len(time_horizons)} time horizons simulation")
    print(f"Total simulations: {len(strategies) * len(time_horizons) * args.n_simulations}")


def compare_with_optimal(args):
    """Compare with global optimal solution"""
    TerminalDisplay.print_header("Strategy vs Global Optimal Solution Comparison", width=70)
    
    # Run strategy simulation
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    strategy_results = runner.run_strategy_comparison(
        strategies=args.strategies or ["upfront_deployment", "gradual_deployment", "flexible_deployment"],
        T=args.time_horizon,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=False,
        show_progress=True
    )
    
    # Run global optimal solution
    TerminalDisplay.print_section("Computing Global Optimal Solution Benchmark")
    
    try:
        from optimal.optimal_solu import main as run_optimal
        import io
        import contextlib
        
        # Temporarily modify T value in parameters file
        original_T = params['economics']['T']
        params['economics']['T'] = args.time_horizon
        
        # Save temporary parameters file
        temp_params_file = project_root / "data" / "parameters_temp.json"
        with open(temp_params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Run global optimal solution (capture output)
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Need to modify test_fixed_model.py to return results instead of just printing
            pass
        
        # Restore original parameters
        params['economics']['T'] = original_T
        temp_params_file.unlink(missing_ok=True)
        
        print("Global optimal solution computation complete")
        
        # Display comparison results
        TerminalDisplay.print_section("Strategy Efficiency Analysis")
        
        for strategy_name, (results, stats) in strategy_results.items():
            efficiency = stats['npv_mean'] / 7000000  # Assume optimal solution is 1M GBP (7M RMB)
            print(f"{strategy_name.title()}: Relative efficiency {efficiency:.1%}")
    
    except Exception as e:
        print(f"Unable to run global optimal solution comparison: {e}")
        print("Please ensure test_fixed_model.py can run normally")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ISRU Strategy Simulation System')
    
    # 添加全局参数（用于默认运行模式）
    parser.add_argument('--visualize', action='store_true',
                       help='Display visualization charts (default: False)')
    parser.add_argument('--time-horizon', type=int, default=10,
                       help='Time horizon (years) (default: 10)')
    parser.add_argument('--n-simulations', type=int, default=50,
                       help='Monte Carlo simulation count (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    def add_common_args(subparser):
        subparser.add_argument('--seed', type=int, default=42, help='Random seed')
        subparser.add_argument('--save', action='store_true', help='Save results')
        subparser.add_argument('--visualize', action='store_true', help='Display visualization charts')
    
    # Single simulation
    single_parser = subparsers.add_parser('single', help='Run single simulation')
    single_parser.add_argument('--strategy', choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                              default='gradual_deployment', help='Strategy type')
    single_parser.add_argument('--time-horizon', type=int, default=30, help='Time horizon')
    add_common_args(single_parser)
    
    # Monte Carlo simulation
    mc_parser = subparsers.add_parser('monte-carlo', help='Run Monte Carlo simulation')
    mc_parser.add_argument('--strategy', choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                          default='gradual_deployment', help='Strategy type')
    mc_parser.add_argument('--time-horizon', type=int, default=30, help='Time horizon')
    mc_parser.add_argument('--n-simulations', type=int, default=100, help='Number of simulations')
    add_common_args(mc_parser)
    
    # Strategy comparison
    compare_parser = subparsers.add_parser('compare', help='Run strategy comparison')
    compare_parser.add_argument('--strategies', nargs='+',
                               choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                               help='Strategies to compare')
    compare_parser.add_argument('--time-horizon', type=int, default=30, help='Time horizon')
    compare_parser.add_argument('--n-simulations', type=int, default=100, help='Number of simulations')
    compare_parser.add_argument('--detailed-analysis', action='store_true', help='Detailed analysis')
    add_common_args(compare_parser)
    
    # Time horizon analysis
    horizon_parser = subparsers.add_parser('horizon', help='Run time horizon analysis')
    horizon_parser.add_argument('--time-horizons', nargs='+', type=int,
                               default=[10, 20, 30, 40, 50], help='Time horizon list')
    horizon_parser.add_argument('--strategies', nargs='+',
                               choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                               help='Strategies to analyze')
    horizon_parser.add_argument('--n-simulations', type=int, default=100, help='Number of simulations')
    horizon_parser.add_argument('--detailed-analysis', action='store_true', help='Detailed analysis')
    add_common_args(horizon_parser)
    
    # Parallel batch simulation
    parallel_parser = subparsers.add_parser('parallel', help='Run parallel batch simulation')
    parallel_parser.add_argument('--time-horizons', nargs='+', type=int,
                                default=[10, 20, 30, 40, 50], help='Time horizon list')
    parallel_parser.add_argument('--strategies', nargs='+',
                                choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                                help='Strategies to analyze')
    parallel_parser.add_argument('--n-simulations', type=int, default=100, help='Number of simulations')
    parallel_parser.add_argument('--max-workers', type=int, help='Maximum parallel processes')
    add_common_args(parallel_parser)
    
    # Compare with optimal solution
    optimal_parser = subparsers.add_parser('optimal', help='Compare with global optimal solution')
    optimal_parser.add_argument('--strategies', nargs='+',
                               choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                               help='Strategies to compare')
    optimal_parser.add_argument('--time-horizon', type=int, default=30, help='Time horizon')
    optimal_parser.add_argument('--n-simulations', type=int, default=100, help='Number of simulations')
    add_common_args(optimal_parser)
    
    # Results management
    results_parser = subparsers.add_parser('results', help='Results management')
    results_subparsers = results_parser.add_subparsers(dest='results_command', help='Results management commands')
    
    # Export to Excel
    export_parser = results_subparsers.add_parser('export', help='Export results to Excel')
    export_parser.add_argument('--strategies', nargs='+',
                              choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                              help='Strategies to export')
    export_parser.add_argument('--time-horizons', nargs='+', type=int,
                              help='Time horizons to export')
    export_parser.add_argument('--output', type=str, help='Output file path')
    
    # View available results
    list_parser = results_subparsers.add_parser('list', help='View available results')
    
    # Clean old results
    cleanup_parser = results_subparsers.add_parser('cleanup', help='Clean old results')
    cleanup_parser.add_argument('--keep-days', type=int, default=30, help='Days to keep')
    
    # Load results
    load_parser = results_subparsers.add_parser('load', help='Load previous results')
    load_parser.add_argument('--strategy', required=True,
                            choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                            help='Strategy name')
    load_parser.add_argument('--time-horizon', type=int, required=True, help='Time horizon')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Display visualization charts for existing results')
    viz_parser.add_argument('--results-dir', type=str, default="strategies/simulation_results",
                           help='Results directory path')
    viz_parser.add_argument('--time-horizon', type=int, default=50,
                           help='Time horizon (years)')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        run_single_simulation(args)
        # Visualization after single simulation
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'monte-carlo':
        run_monte_carlo_simulation(args)
        # Visualization after Monte Carlo simulation
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'compare':
        run_strategy_comparison(args)
        # Visualization after strategy comparison
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'horizon':
        run_time_horizon_analysis(args)
        # Visualization after time horizon analysis
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'parallel':
        run_parallel_batch(args)
        # Visualization after parallel simulation
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'optimal':
        compare_with_optimal(args)
    elif args.command == 'results':
        handle_results_command(args)
    elif args.command == 'visualize':
        # Pure visualization command
        show_visualization(args.results_dir, time_horizon=args.time_horizon)
    else:
        # Default run T=10 strategy comparison with visualization
        run_default_simulation_with_visualization(args)


def handle_results_command(args):
    """Handle results management commands"""
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    if args.results_command == 'export':
        TerminalDisplay.print_header("Export Results to Excel", width=70)
        
        strategies = args.strategies or ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
        time_horizons = args.time_horizons or [10, 20, 30, 40, 50]
        output_file = Path(args.output) if args.output else None
        
        try:
            excel_file = runner.export_results_to_excel(strategies, time_horizons, output_file)
            
            summary_data = {
                "Exported Strategies": ", ".join(strategies),
                "Time Horizons": ", ".join(map(str, time_horizons)),
                "Output File": str(excel_file),
                "File Size": f"{excel_file.stat().st_size / 1024:.1f} KB"
            }
            
            TerminalDisplay.print_summary_box("Export Complete", summary_data, 'green')
            
        except Exception as e:
            print(f"ERROR: Export failed: {e}")
    
    elif args.results_command == 'list':
        TerminalDisplay.print_header("Available Results List", width=70)
        
        available = runner.get_available_results()
        
        if not available:
            print("INFO: No available results")
            print("Please run simulation to generate results first:")
            print("  python strategies/main.py compare --save")
        else:
            for time_horizon, strategies in available.items():
                TerminalDisplay.print_section(f"{time_horizon} Time Horizon")
                for strategy in strategies:
                    print(f"  - {strategy.title()} Strategy")
    
    elif args.results_command == 'cleanup':
        TerminalDisplay.print_header("Clean Old Results", width=70)
        
        print(f"Cleaning result files older than {args.keep_days} days...")
        runner.cleanup_old_results(args.keep_days)
        print("SUCCESS: Cleanup complete")
    
    elif args.results_command == 'load':
        TerminalDisplay.print_header("Load Historical Results", width=70)
        
        result_data = runner.load_previous_results(args.strategy, args.time_horizon)
        
        if result_data:
            metadata = result_data.get('metadata', {})
            results = result_data.get('results', [])
            
            summary_data = {
                "Strategy": args.strategy.title(),
                "Time Horizon": f"{args.time_horizon} years",
                "Simulations": len(results),
                "Generated Time": metadata.get('timestamp', 'Unknown'),
                "Version": metadata.get('version', 'Unknown')
            }
            
            TerminalDisplay.print_summary_box("Result Information", summary_data)
            
            if results:
                # Display simple statistics
                npvs = [r['performance_metrics']['npv'] for r in results]
                npvs_gbp = [npv/7 for npv in npvs]  # Convert to GBP
                print(f"\nNPV Statistics:")
                print(f"  Mean: £{np.mean(npvs_gbp):,.0f}")
                print(f"  Std Dev: £{np.std(npvs_gbp):,.0f}")
                print(f"  Min: £{min(npvs_gbp):,.0f}")
                print(f"  Max: £{max(npvs_gbp):,.0f}")
        else:
            print(f"ERROR: Results not found for {args.strategy} strategy at T={args.time_horizon}")
            print("Available results:")
            available = runner.get_available_results()
            for th, strategies in available.items():
                print(f"  {th}: {', '.join(strategies)}")
    
    else:
        print("Please specify results management command: export, list, cleanup, load")


if __name__ == "__main__":
    main()