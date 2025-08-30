#!/usr/bin/env python3
"""
Batch Simulation Runner
Support multi-strategy, multi-time-horizon batch simulation and comparative analysis
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.simulation_engine import StrategySimulationEngine, SimulationResult
from strategies.core.strategy_definitions import StrategyDefinitions
from strategies.utils.terminal_display import TerminalDisplay
from strategies.utils.result_manager import ResultManager


class BatchSimulationRunner:
    """Batch simulation runner"""
    
    def __init__(self, params: Dict, results_dir: Optional[Path] = None):
        """
        Initialize batch simulation runner
        
        Args:
            params: System parameters
            results_dir: Results save directory
        """
        self.params = params
        self.results_dir = results_dir or Path(__file__).parent.parent / "simulation_results"
        
        # Create simulation engine and result manager
        self.engine = StrategySimulationEngine(params)
        self.result_manager = ResultManager(self.results_dir)
        
        # Default configuration
        self.default_strategies = ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
        self.default_time_horizons = [10, 20, 30, 40, 50]
        self.default_n_simulations = 100
    
    def run_single_batch(self, 
                        strategy_name: str, 
                        T: int, 
                        n_simulations: int = 100,
                        base_seed: int = 42,
                        save_results: bool = True,
                        show_progress: bool = True) -> Tuple[List[SimulationResult], Dict[str, float]]:
        """
        Run batch simulation for single strategy
        
        Args:
            strategy_name: Strategy name
            T: Time horizon
            n_simulations: Number of simulations
            base_seed: Base random seed
            save_results: Whether to save results
            show_progress: Whether to show progress
            
        Returns:
            (simulation results list, statistics)
        """
        if show_progress:
            TerminalDisplay.print_section(f"Running {strategy_name.title()} Strategy (T={T})")
        
        results = []
        
        # Run simulations
        for i in range(n_simulations):
            if show_progress and i % 10 == 0:
                TerminalDisplay.print_simulation_status(strategy_name, T, i, n_simulations)
            
            seed = base_seed + i
            result = self.engine.run_single_simulation(strategy_name, T, seed=seed)
            results.append(result)
        
        if show_progress:
            TerminalDisplay.print_simulation_status(strategy_name, T, n_simulations, n_simulations)
        
        # Calculate statistics
        stats = self.engine.calculate_strategy_statistics(results)
        
        # Save results
        if save_results:
            metadata = {
                'n_simulations': n_simulations,
                'base_seed': base_seed,
                'simulation_type': 'single_batch'
            }
            saved_files = self.result_manager.save_simulation_batch(
                strategy_name, T, results, stats, metadata
            )
            if show_progress:
                print(f"Results saved: {len(saved_files)} files")
        
        return results, stats
    
    def run_strategy_comparison(self, 
                              strategies: Optional[List[str]] = None,
                              T: int = 30,
                              n_simulations: int = 100,
                              base_seed: int = 42,
                              save_results: bool = True,
                              show_progress: bool = True) -> Dict[str, Tuple[List[SimulationResult], Dict[str, float]]]:
        """
        Run strategy comparison simulation
        
        Args:
            strategies: Strategy list
            T: Time horizon
            n_simulations: Number of simulations per strategy
            base_seed: Base random seed
            save_results: Whether to save results
            show_progress: Whether to show progress
            
        Returns:
            Strategy comparison results
        """
        if strategies is None:
            strategies = self.default_strategies
        
        if show_progress:
            TerminalDisplay.print_header(f"Strategy Comparison Simulation (T={T} years)", width=70)
        
        comparison_results = {}
        
        for strategy in strategies:
            results, stats = self.run_single_batch(
                strategy, T, n_simulations, base_seed, save_results, show_progress
            )
            comparison_results[strategy] = (results, stats)
        
        # Display comparison results
        if show_progress:
            self._display_comparison_results(comparison_results, T)
        
        # Save comparison results
        if save_results:
            self.result_manager.save_comparison_results(comparison_results, T, "strategy_comparison")
        
        return comparison_results
    
    def run_time_horizon_analysis(self, 
                                 time_horizons: Optional[List[int]] = None,
                                 strategies: Optional[List[str]] = None,
                                 n_simulations: int = 100,
                                 base_seed: int = 42,
                                 save_results: bool = True,
                                 show_progress: bool = True) -> Dict[int, Dict[str, Tuple[List[SimulationResult], Dict[str, float]]]]:
        """
        Run time horizon analysis
        
        Args:
            time_horizons: Time horizon list
            strategies: Strategy list
            n_simulations: Number of simulations per strategy
            base_seed: Base random seed
            save_results: Whether to save results
            show_progress: Whether to show progress
            
        Returns:
            Time horizon analysis results
        """
        if time_horizons is None:
            time_horizons = self.default_time_horizons
        if strategies is None:
            strategies = self.default_strategies
        
        if show_progress:
            TerminalDisplay.print_header("Time Horizon Analysis", width=70)
        
        horizon_results = {}
        
        for T in time_horizons:
            if show_progress:
                TerminalDisplay.print_section(f"Time Horizon: {T} years")
            
            comparison = self.run_strategy_comparison(
                strategies, T, n_simulations, base_seed, save_results, False
            )
            horizon_results[T] = comparison
        
        # Display time horizon analysis results
        if show_progress:
            self._display_horizon_analysis(horizon_results)
        
        # Save time horizon analysis results
        if save_results:
            self.result_manager.save_horizon_analysis(horizon_results, strategies)
        
        return horizon_results
    
    def run_parallel_batch(self, 
                          strategies: Optional[List[str]] = None,
                          time_horizons: Optional[List[int]] = None,
                          n_simulations: int = 100,
                          base_seed: int = 42,
                          max_workers: Optional[int] = None,
                          save_results: bool = True) -> Dict[int, Dict[str, Tuple[List[SimulationResult], Dict[str, float]]]]:
        """
        Run parallel batch simulation
        
        Args:
            strategies: Strategy list
            time_horizons: Time horizon list
            n_simulations: Number of simulations per strategy
            base_seed: Base random seed
            max_workers: Maximum parallel worker processes
            save_results: Whether to save results
            
        Returns:
            Parallel simulation results
        """
        if strategies is None:
            strategies = self.default_strategies
        if time_horizons is None:
            time_horizons = self.default_time_horizons
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(strategies) * len(time_horizons))
        
        TerminalDisplay.print_header("Parallel Batch Simulation", width=70)
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Time Horizons: {time_horizons}")
        print(f"Simulations: {n_simulations}/strategy")
        print(f"Parallel Processes: {max_workers}")
        print()
        
        # Prepare task list
        tasks = []
        for T in time_horizons:
            for strategy in strategies:
                tasks.append((strategy, T, n_simulations, base_seed))
        
        # Parallel execution
        results = {}
        completed_tasks = 0
        total_tasks = len(tasks)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_task, task): task
                for task in tasks
            }
            
            # Collect results
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                strategy, T, _, _ = task
                
                try:
                    task_results, task_stats = future.result()
                    
                    # Organize results
                    if T not in results:
                        results[T] = {}
                    results[T][strategy] = (task_results, task_stats)
                    
                    # Save results
                    if save_results:
                        metadata = {
                            'n_simulations': n_simulations,
                            'base_seed': base_seed,
                            'simulation_type': 'parallel_batch'
                        }
                        self.result_manager.save_simulation_batch(
                            strategy, T, task_results, task_stats, metadata
                        )
                    
                    completed_tasks += 1
                    TerminalDisplay.print_progress_bar(
                        completed_tasks, total_tasks,
                        prefix="Overall Progress",
                        suffix=f"{completed_tasks}/{total_tasks} tasks completed"
                    )
                    
                except Exception as e:
                    print(f"Task {task} execution failed: {e}")
        
        print("\nParallel simulation completed!")
        return results
    
    def _run_single_task(self, task: Tuple[str, int, int, int]) -> Tuple[List[SimulationResult], Dict[str, float]]:
        """Run single simulation task (for parallel execution)"""
        strategy, T, n_simulations, base_seed = task
        
        # Create new engine instance in subprocess
        engine = StrategySimulationEngine(self.params)
        results = []
        
        for i in range(n_simulations):
            seed = base_seed + i
            result = engine.run_single_simulation(strategy, T, seed=seed)
            results.append(result)
        
        stats = engine.calculate_strategy_statistics(results)
        return results, stats
    
    def export_results_to_excel(self,
                               strategies: Optional[List[str]] = None,
                               time_horizons: Optional[List[int]] = None,
                               output_file: Optional[Path] = None) -> Path:
        """
        Export results to Excel file
        
        Args:
            strategies: Strategy list
            time_horizons: Time horizon list
            output_file: Output file path
            
        Returns:
            Excel file path
        """
        if strategies is None:
            strategies = self.default_strategies
        if time_horizons is None:
            time_horizons = self.default_time_horizons
        
        return self.result_manager.export_to_excel(strategies, time_horizons, output_file)
    
    def get_available_results(self) -> Dict[str, List[str]]:
        """Get available results list"""
        return self.result_manager.get_available_results()
    
    def load_previous_results(self, strategy_name: str, T: int) -> Optional[Dict]:
        """Load previous simulation results"""
        return self.result_manager.load_simulation_results(strategy_name, T)
    
    def cleanup_old_results(self, keep_days: int = 30):
        """Clean up old result files"""
        self.result_manager.cleanup_old_results(keep_days)
    
    def _display_comparison_results(self, comparison_results: Dict, T: int):
        """Display strategy comparison results"""
        TerminalDisplay.print_section(f"T={T} Years Strategy Comparison Results")
        
        # Prepare comparison data
        comparison_data = {}
        for strategy, (results, stats) in comparison_results.items():
            comparison_data[strategy] = {
                'npv_mean': stats['npv_mean'],
                'npv_std': stats['npv_std'],
                'utilization_mean': stats['utilization_mean'],
                'self_sufficiency_mean': stats['self_sufficiency_mean'],
                'probability_positive_npv': stats['probability_positive_npv']
            }
        
        # Display comparison table
        TerminalDisplay.print_comparison_table(comparison_data, f"T={T} Years Strategy Comparison")
        
        # Display best strategy
        best_strategy = max(comparison_data.keys(),
                          key=lambda s: comparison_data[s]['npv_mean'])
        
        best_data = {
            "Best Strategy": best_strategy.title(),
            "NPV Mean": comparison_data[best_strategy]['npv_mean'],
            "Utilization": comparison_data[best_strategy]['utilization_mean'],
            "Success Rate": comparison_data[best_strategy]['probability_positive_npv']
        }
        
        TerminalDisplay.print_summary_box(f"T={T} Years Best Strategy", best_data, 'green')
    
    def _display_horizon_analysis(self, horizon_results: Dict):
        """Display time horizon analysis results"""
        TerminalDisplay.print_section("Time Horizon Analysis Summary")
        
        # Prepare analysis data
        analysis_data = []
        for T, strategies in horizon_results.items():
            for strategy, (results, stats) in strategies.items():
                analysis_data.append({
                    "Time Horizon": T,
                    "Strategy": strategy.title(),
                    "NPV Mean": stats['npv_mean'],
                    "NPV Std": stats['npv_std'],
                    "Utilization": stats['utilization_mean'],
                    "Success Rate": stats['probability_positive_npv']
                })
        
        # Define table columns
        from strategies.utils.terminal_display import TableColumn
        columns = [
            TableColumn("Time Horizon", 8, 'center'),
            TableColumn("Strategy", 12, 'left'),
            TableColumn("NPV Mean", 12, 'right', TerminalDisplay._format_number),
            TableColumn("NPV Std", 12, 'right', TerminalDisplay._format_number),
            TableColumn("Utilization", 10, 'right', lambda x: f"{x:.1%}"),
            TableColumn("Success Rate", 10, 'right', lambda x: f"{x:.1%}")
        ]
        
        TerminalDisplay.print_table(analysis_data, columns, "Time Horizon Analysis Results")


def load_parameters() -> Dict:
    """Load system parameters"""
    params_path = project_root / "data" / "parameters.json"
    with open(params_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test code
    print("=== Batch Simulation Runner Test ===")
    
    # Load parameters
    params = load_parameters()
    
    # Create batch runner
    runner = BatchSimulationRunner(params)
    
    # Test single strategy simulation
    print("\n--- Single Strategy Simulation Test ---")
    results, stats = runner.run_single_batch("upfront_deployment", T=10, n_simulations=20)
    print(f"Completed {len(results)} simulations")
    print(f"NPV Mean: ${stats['npv_mean']:,.0f}")
    
    # Test strategy comparison
    print("\n--- Strategy Comparison Test ---")
    comparison = runner.run_strategy_comparison(
        strategies=["upfront_deployment", "flexible_deployment"],
        T=10,
        n_simulations=10
    )
    
    # Test time horizon analysis
    print("\n--- Time Horizon Analysis Test ---")
    horizon_analysis = runner.run_time_horizon_analysis(
        time_horizons=[10, 20],
        strategies=["gradual_deployment"],
        n_simulations=5
    )