#!/usr/bin/env python3
"""
Strategy Simulation Engine - Core Simulator
Rule-driven strategy execution, not optimization solving
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from dataclasses import asdict

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.strategy_definitions import StrategyDefinitions, StrategyParams, StrategyType
from strategies.core.state_manager import SystemState, StateManager, Decision
from strategies.core.decision_logic import DecisionEngine
from analysis.gbm_demand import GBMDemandGenerator


class SimulationResult:
    """Simulation result data structure"""
    
    def __init__(self, strategy_name: str, T: int):
        self.strategy_name = strategy_name
        self.T = T
        self.demand_path: List[float] = []
        self.decisions: List[Decision] = []
        self.states: List[SystemState] = []
        self.performance_metrics: Dict[str, float] = {}
        self.simulation_params: Dict = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for serialization"""
        return {
            'strategy_name': self.strategy_name,
            'T': self.T,
            'demand_path': self.demand_path,
            'decisions': [asdict(d) for d in self.decisions],
            'states': [asdict(s) for s in self.states],
            'performance_metrics': self.performance_metrics,
            'simulation_params': self.simulation_params
        }


class StrategySimulationEngine:
    """Strategy simulation engine"""
    
    def __init__(self, params: Dict):
        """
        Initialize simulation engine
        
        Args:
            params: System parameters dictionary
        """
        self.params = params
        self.demand_generator = self._create_demand_generator()
        
    def _create_demand_generator(self) -> GBMDemandGenerator:
        """Create demand generator"""
        demand_params = self.params['demand']
        return GBMDemandGenerator(
            D0=demand_params['D0'],
            mu=demand_params['mu'],
            sigma=demand_params['sigma'],
            dt=demand_params['dt']
        )
    
    def run_single_simulation(self,
                            strategy_name: str,
                            T: int,
                            seed: Optional[int] = None,
                            demand_path: Optional[List[float]] = None) -> SimulationResult:
        """
        Run single strategy simulation
        
        Args:
            strategy_name: Strategy name
            T: Time horizon
            seed: Random seed
            demand_path: Predefined demand path (optional)
            
        Returns:
            Simulation result
        """
        # Get strategy parameters
        strategies = StrategyDefinitions.get_all_strategies(T)
        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = strategies[strategy_name]
        
        # Generate or use demand path
        if demand_path is None:
            demand_path = self.demand_generator.generate_single_path(T, seed=seed)
        
        # Ensure demand path length is correct
        if len(demand_path) != T + 1:
            # Adjust length, remove t=0 initial point
            demand_path = demand_path[1:T+1] if len(demand_path) > T else demand_path[:T]
        
        # Initialize simulation components
        decision_engine = DecisionEngine(strategy, self.params)
        
        # New strategies don't need initial deployment, all deployments handled by decision logic
        initial_capacity = 0.0
        initial_deployed_mass = 0.0
        
        # Create initial state
        initial_state = SystemState(
            current_time=0,
            total_capacity=initial_capacity,
            deployed_mass=initial_deployed_mass,
            current_demand=float(demand_path[0]) if len(demand_path) > 0 else 0.0,
            inventory=0.0
        )
        
        state_manager = StateManager(initial_state)
        
        # Create result object
        result = SimulationResult(strategy_name, T)
        result.demand_path = demand_path
        result.simulation_params = {
            'initial_capacity': initial_capacity,
            'initial_deployed_mass': initial_deployed_mass,
            'strategy_params': asdict(strategy),
            'seed': seed
        }
        
        # Execute simulation loop
        for t in range(T):
            current_demand = float(demand_path[t]) if t < len(demand_path) else 0.0
            
            # Get demand forecast (simple linear prediction)
            demand_forecast = self._generate_demand_forecast(demand_path, t, forecast_horizon=3)
            
            # Make decision
            decision = decision_engine.make_decision(state_manager.state, demand_forecast)
            
            # Update state
            new_state = state_manager.update_state(decision, current_demand, self.params)
            
            # Record results
            result.decisions.append(decision)
            result.states.append(SystemState(**asdict(new_state)))  # Create copy
        
        # Calculate performance metrics
        result.performance_metrics = state_manager.calculate_performance_metrics(self.params)
        
        return result
    
    def _generate_demand_forecast(self, demand_path: List[float],
                                current_time: int,
                                forecast_horizon: int = 3) -> List[float]:
        """
        Generate simple demand forecast
        
        Args:
            demand_path: Complete demand path
            current_time: Current time
            forecast_horizon: Forecast time horizon
            
        Returns:
            Predicted demand sequence
        """
        if current_time >= len(demand_path) - 1:
            return []
        
        # Simple linear trend prediction
        available_future = list(demand_path[current_time + 1:current_time + 1 + forecast_horizon])
        
        if len(available_future) < forecast_horizon and current_time > 0:
            # If future data insufficient, predict based on historical trends
            recent_demands = demand_path[max(0, current_time - 2):current_time + 1]
            if len(recent_demands) >= 2:
                growth_rate = (recent_demands[-1] / recent_demands[0]) ** (1 / (len(recent_demands) - 1)) - 1
                last_demand = demand_path[current_time]
                
                # Supplement predicted values
                for i in range(len(available_future), forecast_horizon):
                    predicted_demand = last_demand * ((1 + growth_rate) ** (i + 1))
                    available_future.append(predicted_demand)
        
        return available_future[:forecast_horizon]
    
    def run_monte_carlo_simulation(self,
                                 strategy_name: str,
                                 T: int,
                                 n_simulations: int = 100,
                                 base_seed: int = 42) -> List[SimulationResult]:
        """
        Run Monte Carlo simulation
        
        Args:
            strategy_name: Strategy name
            T: Time horizon
            n_simulations: Number of simulations
            base_seed: Base random seed
            
        Returns:
            List of simulation results
        """
        results = []
        
        for i in range(n_simulations):
            seed = base_seed + i
            result = self.run_single_simulation(strategy_name, T, seed=seed)
            results.append(result)
        
        return results
    
    def compare_strategies(self,
                         strategy_names: List[str],
                         T: int,
                         n_simulations: int = 100,
                         base_seed: int = 42) -> Dict[str, List[SimulationResult]]:
        """
        Compare multiple strategies
        
        Args:
            strategy_names: List of strategy names
            T: Time horizon
            n_simulations: Number of simulations per strategy
            base_seed: Base random seed
            
        Returns:
            Strategy comparison results
        """
        comparison_results = {}
        
        for strategy_name in strategy_names:
            print(f"Simulating {strategy_name} strategy...")
            results = self.run_monte_carlo_simulation(strategy_name, T, n_simulations, base_seed)
            comparison_results[strategy_name] = results
        
        return comparison_results
    
    def calculate_strategy_statistics(self, results: List[SimulationResult]) -> Dict[str, float]:
        """
        Calculate strategy statistics
        
        Args:
            results: List of simulation results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        # Extract key metrics
        npvs = [r.performance_metrics.get('npv', 0) for r in results]
        utilizations = [r.performance_metrics.get('avg_utilization', 0) for r in results]
        self_sufficiency_rates = [r.performance_metrics.get('self_sufficiency_rate', 0) for r in results]
        total_costs = [r.performance_metrics.get('total_cost', 0) for r in results]
        
        return {
            # NPV statistics
            'npv_mean': float(np.mean(npvs)),
            'npv_std': float(np.std(npvs)),
            'npv_min': float(np.min(npvs)),
            'npv_max': float(np.max(npvs)),
            'npv_p5': float(np.percentile(npvs, 5)),
            'npv_p95': float(np.percentile(npvs, 95)),
            
            # Utilization statistics
            'utilization_mean': float(np.mean(utilizations)),
            'utilization_std': float(np.std(utilizations)),
            
            # Self-sufficiency statistics
            'self_sufficiency_mean': float(np.mean(self_sufficiency_rates)),
            'self_sufficiency_std': float(np.std(self_sufficiency_rates)),
            
            # Cost statistics
            'total_cost_mean': float(np.mean(total_costs)),
            'total_cost_std': float(np.std(total_costs)),
            
            # Risk indicators
            'npv_coefficient_of_variation': float(np.std(npvs) / np.mean(npvs)) if np.mean(npvs) != 0 else 0,
            'probability_positive_npv': float(np.mean([npv > 0 for npv in npvs])),
            
            # Simulation metadata
            'n_simulations': len(results),
            'strategy_name': results[0].strategy_name if results else 'unknown'
        }


if __name__ == "__main__":
    # Test code
    print("=== Strategy Simulation Engine Test ===")
    
    # Load parameters
    params_path = project_root / "data" / "parameters.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Create simulation engine
    engine = StrategySimulationEngine(params)
    
    # Test single simulation
    print("\n--- Single Simulation Test ---")
    result = engine.run_single_simulation("upfront_deployment", T=10, seed=42)
    print(f"Strategy: {result.strategy_name}")
    print(f"Time Horizon: {result.T}")
    print(f"Final NPV: ${result.performance_metrics.get('npv', 0):,.0f}")
    print(f"Average Utilization: {result.performance_metrics.get('avg_utilization', 0):.1%}")
    print(f"Self-Sufficiency Rate: {result.performance_metrics.get('self_sufficiency_rate', 0):.1%}")
    
    # Test Monte Carlo simulation
    print("\n--- Monte Carlo Simulation Test ---")
    mc_results = engine.run_monte_carlo_simulation("gradual_deployment", T=10, n_simulations=10, base_seed=42)
    stats = engine.calculate_strategy_statistics(mc_results)
    print(f"NPV Mean: ${stats['npv_mean']:,.0f}")
    print(f"NPV Std Dev: ${stats['npv_std']:,.0f}")
    print(f"Average Utilization: {stats['utilization_mean']:.1%}")
    
    # Test strategy comparison
    print("\n--- Strategy Comparison Test ---")
    comparison = engine.compare_strategies(["upfront_deployment", "flexible_deployment"], T=10, n_simulations=5)
    for strategy, results in comparison.items():
        stats = engine.calculate_strategy_statistics(results)
        print(f"{strategy}: NPV=${stats['npv_mean']:,.0f}±${stats['npv_std']:,.0f}")