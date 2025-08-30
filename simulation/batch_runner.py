"""
Batch Simulation Runner
Supports batch simulation with multiple scenarios and parameter combinations
"""

import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import pickle

from models.isru_model import ISRUOptimizationModel
from analysis.gbm_demand import GBMDemandGenerator

logger = logging.getLogger(__name__)


class BatchRunner:
    """
    Batch simulation runner
    
    Supports the following features:
    - Multi-scenario batch simulation
    - Parallel computing
    - Result aggregation and analysis
    - Progress tracking
    """
    
    def __init__(self, params: Dict[str, Any], output_dir: str = "results"):
        """
        Initialize batch runner
        
        Args:
            params: Base parameter dictionary
            output_dir: Output directory
        """
        self.params = params
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_file = os.path.join(self.output_dir, f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_single_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run single scenario
        
        Args:
            scenario: Scenario configuration
            
        Returns:
            Scenario result
        """
        logger.info(f"Running scenario: {scenario.get('name', 'unnamed')}")
        
        try:
            # Merge parameters
            scenario_params = self._merge_params(self.params, scenario.get('params', {}))
            
            # Generate demand path
            demand_generator = GBMDemandGenerator(scenario_params)
            demand_path = demand_generator.generate_path(
                n_scenarios=1,
                random_seed=scenario.get('seed', 42)
            )[0]
            
            # Create and solve model
            model = ISRUOptimizationModel(scenario_params)
            model.build_model(demand_path)
            result = model.solve()
            
            # Add scenario information
            result['scenario'] = scenario
            result['demand_path'] = demand_path
            
            return result
            
        except Exception as e:
            logger.error(f"Scenario execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'scenario': scenario
            }
    
    def run_batch(self, scenarios: List[Dict[str, Any]], 
                  parallel: bool = True, max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Run batch simulation
        
        Args:
            scenarios: Scenario list
            parallel: Whether to run in parallel
            max_workers: Maximum number of worker processes
            
        Returns:
            List of results from all scenarios
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(scenarios))
        
        logger.info(f"Starting batch simulation with {len(scenarios)} scenarios")
        
        if parallel and len(scenarios) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_scenario = {
                    executor.submit(self.run_single_scenario, scenario): scenario
                    for scenario in scenarios
                }
                
                for future in as_completed(future_to_scenario):
                    result = future.result()
                    self.results.append(result)
                    
                    # Save intermediate result
                    self._save_intermediate_result(result)
                    
        else:
            # Serial execution
            for scenario in scenarios:
                result = self.run_single_scenario(scenario)
                self.results.append(result)
                
                # Save intermediate result
                self._save_intermediate_result(result)
        
        logger.info("Batch simulation completed")
        
        # Save complete results
        self._save_all_results()
        
        return self.results
    
    def _merge_params(self, base_params: Dict, override_params: Dict) -> Dict:
        """Merge parameter dictionaries"""
        merged = base_params.copy()
        
        def deep_merge(d1, d2):
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value
        
        deep_merge(merged, override_params)
        return merged
    
    def _save_intermediate_result(self, result: Dict[str, Any]) -> None:
        """Save intermediate result"""
        scenario_name = result['scenario'].get('name', 'unnamed')
        filename = f"result_{scenario_name}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to serializable format
        serializable_result = self._make_serializable(result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
    
    def _save_all_results(self) -> None:
        """Save all results"""
        filename = f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"All results saved to: {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def get_results_summary(self) -> pd.DataFrame:
        """Get results summary"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for result in self.results:
            if result['status'] == 'optimal':
                summary = {
                    'scenario': result['scenario'].get('name', 'unnamed'),
                    'status': result['status'],
                    'objective_value': result['objective_value'],
                    'total_demand': sum(result['demand_path']),
                    'total_supply': sum(result['solution']['Qt'].values()) if 'solution' in result else 0,
                    'total_shortage': sum(result['solution']['St'].values()) if 'solution' in result else 0,
                    'max_capacity': max(result['solution']['Qt_cap'].values()) if 'solution' in result else 0,
                    'total_mass': max(result['solution']['Mt'].values()) if 'solution' in result else 0
                }
                
                # Add scenario parameters
                for key, value in result['scenario'].get('params', {}).items():
                    if isinstance(value, (int, float, str)):
                        summary[f'param_{key}'] = value
                
                summary_data.append(summary)
            else:
                summary = {
                    'scenario': result['scenario'].get('name', 'unnamed'),
                    'status': result['status'],
                    'error': result.get('error', '')
                }
                summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_summary_report(self) -> str:
        """Save summary report"""
        summary_df = self.get_results_summary()
        
        if summary_df.empty:
            logger.warning("No results to save")
            return ""
        
        filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        summary_df.to_csv(filepath, index=False)
        logger.info(f"Summary report saved to: {filepath}")
        
        return filepath


class ScenarioGenerator:
    """Scenario generator"""
    
    @staticmethod
    def generate_parameter_sweep(base_params: Dict, 
                                 param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Generate parameter sweep scenarios
        
        Args:
            base_params: Base parameters
            param_ranges: Parameter range dictionary
            
        Returns:
            Scenario list
        """
        scenarios = []
        
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        from itertools import product
        
        for i, combination in enumerate(product(*param_values)):
            scenario = {
                'name': f'param_sweep_{i}',
                'params': {}
            }
            
            # Build parameter hierarchy
            for name, value in zip(param_names, combination):
                keys = name.split('.')
                current = scenario['params']
                
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                current[keys[-1]] = value
            
            scenarios.append(scenario)
        
        return scenarios
    
    @staticmethod
    def generate_demand_scenarios(base_params: Dict, 
                                  demand_configs: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate demand scenarios
        
        Args:
            base_params: Base parameters
            demand_configs: Demand configuration list
            
        Returns:
            Scenario list
        """
        scenarios = []
        
        for i, config in enumerate(demand_configs):
            scenario = {
                'name': f'demand_scenario_{i}',
                'params': {
                    'demand': config
                },
                'seed': config.get('seed', 42 + i)
            }
            scenarios.append(scenario)
        
        return scenarios
    
    @staticmethod
    def generate_uncertainty_scenarios(base_params: Dict, 
                                       uncertainty_factors: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Generate uncertainty scenarios
        
        Args:
            base_params: Base parameters
            uncertainty_factors: Uncertainty factors
            
        Returns:
            Scenario list
        """
        scenarios = []
        
        # Generate uncertainty combinations
        factor_names = list(uncertainty_factors.keys())
        factor_values = [uncertainty_factors[name] for name in factor_names]
        
        from itertools import product
        
        for i, combination in enumerate(product(*factor_values)):
            scenario = {
                'name': f'uncertainty_{i}',
                'params': {
                    'costs': {},
                    'technology': {},
                    'economics': {}
                }
            }
            
            # Apply uncertainty factors
            for name, value in zip(factor_names, combination):
                if name.startswith('cost_'):
                    scenario['params']['costs'][name[5:]] = base_params['costs'][name[5:]] * value
                elif name.startswith('tech_'):
                    scenario['params']['technology'][name[5:]] = base_params['technology'][name[5:]] * value
                elif name.startswith('econ_'):
                    scenario['params']['economics'][name[5:]] = base_params['economics'][name[5:]] * value
            
            scenarios.append(scenario)
        
        return scenarios


# Convenience functions
def run_parameter_sweep(params: Dict, param_ranges: Dict,
                        output_dir: str = "results") -> pd.DataFrame:
    """
    Run parameter sweep
    
    Args:
        params: Base parameters
        param_ranges: Parameter ranges
        output_dir: Output directory
        
    Returns:
        Results summary DataFrame
    """
    generator = ScenarioGenerator()
    scenarios = generator.generate_parameter_sweep(params, param_ranges)
    
    runner = BatchRunner(params, output_dir)
    results = runner.run_batch(scenarios)
    
    return runner.get_results_summary()


if __name__ == "__main__":
    # Test batch runner
    import json
    
    # Load parameters
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    
    # Create test scenarios
    scenarios = [
        {
            'name': 'baseline',
            'params': {}
        },
        {
            'name': 'high_demand',
            'params': {
                'demand': {'D0': params['demand']['D0'] * 1.5}
            }
        },
        {
            'name': 'low_cost',
            'params': {
                'costs': {'c_op': params['costs']['c_op'] * 0.8}
            }
        }
    ]
    
    # Run batch simulation
    runner = BatchRunner(params)
    results = runner.run_batch(scenarios, parallel=False)
    
    # Save summary
    summary_file = runner.save_summary_report()
    print(f"Results summary saved to: {summary_file}")