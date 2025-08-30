#!/usr/bin/env python3
"""
Decision Logic Engine - New Strategy Version
Implement specific decision logic for three new deployment strategies
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.strategy_definitions import StrategyParams, StrategyType, calculate_deployment_for_year
from strategies.core.state_manager import SystemState, Decision


class DecisionEngine:
    """New strategy decision engine"""
    
    def __init__(self, strategy: StrategyParams, params: Dict):
        """
        Initialize decision engine
        
        Args:
            strategy: Strategy parameters
            params: System parameters
        """
        self.strategy = strategy
        self.params = params
        
        # Extract cost information from parameters
        costs = params.get('costs', {})
        self.capacity_cost_per_kg = costs.get('c_dev', 10000)  # Capacity construction cost
        self.operational_cost_per_kg = costs.get('c_op', 3000)  # Operational cost
        self.earth_supply_cost_per_kg = costs.get('c_E', 20000)  # Earth supply cost
        
        # Technology parameters
        tech = params.get('technology', {})
        self.efficiency_ratio = tech.get('eta', 2)  # Capacity efficiency ratio
        
        # Demand parameters
        demand_params = params.get('demand', {})
        self.D0 = demand_params.get('D0', 10)  # Initial demand
        self.mu = demand_params.get('mu', 0.2)  # Demand growth rate
        self.sigma = demand_params.get('sigma', 0.2)  # Demand volatility
    
    def make_decision(self, state: SystemState, demand_forecast: Optional[List[float]] = None) -> Decision:
        """
        Make decisions based on new strategy and current state
        
        Args:
            state: Current system state
            demand_forecast: Demand forecast (optional)
            
        Returns:
            Decision result
        """
        decision = Decision(time=state.current_time)
        
        # 1. Calculate final demand (based on time horizon T)
        final_demand = self._estimate_final_demand(state)
        
        # 2. Calculate current year deployment based on strategy
        deployment_amount = self._calculate_deployment_amount(state, final_demand)
        
        if deployment_amount > 0:
            decision.capacity_expansion = deployment_amount
            decision.new_deployment = deployment_amount / self.efficiency_ratio
            decision.expansion_cost = self._calculate_expansion_cost(deployment_amount)
            decision.decision_reason = f"{self.strategy.deployment_type} strategy deployment"
        
        # 3. Calculate production plan
        decision.planned_production = self._calculate_production_plan(state)
        decision.operational_cost = self._calculate_operational_cost(decision.planned_production)
        
        # 4. Calculate earth supply requirements
        production_shortfall = max(0, state.current_demand - decision.planned_production)
        decision.earth_supply_request = production_shortfall
        decision.supply_cost = self._calculate_supply_cost(decision.earth_supply_request)
        
        # 5. Set inventory target
        decision.inventory_target = self._calculate_inventory_target(state)
        
        return decision
    
    def _estimate_final_demand(self, state: SystemState) -> float:
        """
        Estimate final year demand
        
        Args:
            state: Current system state
            
        Returns:
            Expected demand in final year
        """
        # Use geometric Brownian motion model to estimate final demand
        # D_T = D0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        # Here use expected value: D_T = D0 * exp(mu*T)
        
        T = self.strategy.time_horizon
        final_demand = self.D0 * np.exp(self.mu * T)
        
        return final_demand
    
    def _calculate_deployment_amount(self, state: SystemState, final_demand: float) -> float:
        """
        Calculate current year deployment based on strategy
        
        Args:
            state: Current system state
            final_demand: Final demand
            
        Returns:
            New deployment amount for current year
        """
        current_year = state.current_time
        
        # Get previous year supply-demand information (for flexible strategy)
        previous_supply = 0
        previous_demand = 0
        
        if len(state.demand_history) > 0 and len(state.capacity_history) > 0:
            previous_demand = state.demand_history[-1]
            previous_capacity = state.capacity_history[-1] if state.capacity_history else 0
            # Assume previous year supply equals capacity (simplified)
            previous_supply = previous_capacity
        
        return calculate_deployment_for_year(
            self.strategy,
            current_year,
            final_demand,
            previous_supply,
            previous_demand
        )
    
    def _calculate_production_plan(self, state: SystemState) -> float:
        """Calculate production plan"""
        # Production plan limited by capacity
        max_production = state.total_capacity
        
        # Adjust production plan based on demand and inventory
        target_production = state.current_demand
        
        # Consider inventory situation
        if state.inventory > state.current_demand * 0.1:  # Inventory exceeds 10% of annual demand
            # Reduce production to avoid inventory buildup
            target_production *= 0.9
        
        return min(target_production, max_production)
    
    def _calculate_inventory_target(self, state: SystemState) -> float:
        """Calculate inventory target"""
        # Base target: 10% of annual demand
        base_target = state.current_demand * 0.1
        
        # Adjust based on strategy type
        if self.strategy.deployment_type == "upfront":
            # Upfront deployment strategy: maintain higher inventory
            return base_target * 1.5
        elif self.strategy.deployment_type == "gradual":
            # Gradual deployment strategy: medium inventory
            return base_target * 1.2
        else:  # flexible
            # Flexible deployment strategy: lower inventory, rely on quick response
            return base_target * 0.8
    
    def _calculate_expansion_cost(self, expansion_amount: float) -> float:
        """Calculate expansion cost"""
        # Capacity construction cost
        capacity_cost = expansion_amount * self.capacity_cost_per_kg
        
        # Economies of scale effect
        if expansion_amount > 100:  # Large-scale expansion gets discount
            capacity_cost *= 0.9
        
        return capacity_cost
    
    def _calculate_operational_cost(self, production_amount: float) -> float:
        """Calculate operational cost"""
        return production_amount * self.operational_cost_per_kg
    
    def _calculate_supply_cost(self, supply_amount: float) -> float:
        """Calculate earth supply cost"""
        return supply_amount * self.earth_supply_cost_per_kg


if __name__ == "__main__":
    # Test code
    print("=== New Strategy Decision Logic Engine Test ===")
    
    # Import strategy definitions
    from strategies.core.strategy_definitions import StrategyDefinitions
    
    # Test parameters
    params = {
        'costs': {
            'c_dev': 10000,
            'c_op': 3000,
            'c_E': 20000
        },
        'technology': {
            'eta': 2
        },
        'demand': {
            'D0': 10,
            'mu': 0.2,
            'sigma': 0.2
        }
    }
    
    # Test each new strategy
    time_horizon = 20
    strategies = StrategyDefinitions.get_all_strategies(time_horizon)
    
    for name, strategy in strategies.items():
        print(f"\n=== {name.upper()} Strategy Test ===")
        
        engine = DecisionEngine(strategy, params)
        
        # Create test state
        test_state = SystemState(
            current_time=5,
            total_capacity=50.0,
            current_demand=60.0,  # Demand exceeds capacity
            inventory=5.0,
            demand_history=[40, 45, 50, 55, 60],
            capacity_history=[30, 35, 40, 45, 50]
        )
        
        # Make decision
        decision = engine.make_decision(test_state)
        
        print(f"Decision Results:")
        print(f"  Capacity Expansion: {decision.capacity_expansion:.1f}")
        print(f"  New Deployment: {decision.new_deployment:.1f}")
        print(f"  Planned Production: {decision.planned_production:.1f}")
        print(f"  Earth Supply: {decision.earth_supply_request:.1f}")
        print(f"  Expansion Cost: ${decision.expansion_cost:,.0f}")
        print(f"  Decision Reason: {decision.decision_reason}")