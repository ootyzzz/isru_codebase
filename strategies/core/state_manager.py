#!/usr/bin/env python3
"""
System State Manager
Manage ISRU system state information and state transitions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class SystemState:
    """System state data structure"""
    # Time information
    current_time: int = 0
    
    # Capacity information
    total_capacity: float = 0.0          # Total capacity [kg/year]
    deployed_mass: float = 0.0           # Deployed ISRU mass [kg]
    
    # Demand and production
    current_demand: float = 0.0          # Current demand [kg]
    actual_production: float = 0.0       # Actual production [kg]
    
    # Inventory management
    inventory: float = 0.0               # Current inventory [kg]
    
    # Supply information
    earth_supply: float = 0.0            # Earth supply amount [kg]
    
    # Historical records
    demand_history: List[float] = field(default_factory=list)
    production_history: List[float] = field(default_factory=list)
    capacity_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.update_derived_metrics()
    
    def update_derived_metrics(self):
        """Update derived metrics"""
        # Utilization rate
        if self.total_capacity > 0:
            self.utilization_rate = min(self.current_demand / self.total_capacity, 1.0)
        else:
            self.utilization_rate = 0.0
        
        # Inventory level (days relative to demand)
        if self.current_demand > 0:
            self.inventory_days = self.inventory / (self.current_demand / 365)
        else:
            self.inventory_days = float('inf') if self.inventory > 0 else 0.0
        
        # Supply-demand balance
        self.supply_demand_ratio = (self.actual_production + self.earth_supply) / max(self.current_demand, 1e-6)
        
        # Self-sufficiency rate
        self.self_sufficiency_rate = self.actual_production / max(self.current_demand, 1e-6)


@dataclass
class Decision:
    """Decision result data structure"""
    time: int
    
    # Capacity decisions
    capacity_expansion: float = 0.0      # New capacity [kg/year]
    new_deployment: float = 0.0          # New deployment mass [kg]
    
    # Production decisions
    planned_production: float = 0.0      # Planned production [kg]
    
    # Supply decisions
    earth_supply_request: float = 0.0    # Earth supply request [kg]
    
    # Inventory decisions
    inventory_target: float = 0.0        # Target inventory [kg]
    
    # Decision reason
    decision_reason: str = ""
    
    # Cost estimates
    expansion_cost: float = 0.0          # Expansion cost
    operational_cost: float = 0.0        # Operational cost
    supply_cost: float = 0.0             # Supply cost


class StateManager:
    """System state manager"""
    
    def __init__(self, initial_state: Optional[SystemState] = None):
        """Initialize state manager"""
        self.state = initial_state or SystemState()
        self.decision_history: List[Decision] = []
        
    def update_state(self, decision: Decision, new_demand: float, params: Dict) -> SystemState:
        """
        Update system state based on decision and new demand
        
        Args:
            decision: Current decision
            new_demand: New demand
            params: System parameters
            
        Returns:
            Updated state
        """
        # Update time
        self.state.current_time += 1
        
        # Update demand
        self.state.current_demand = new_demand
        self.state.demand_history.append(new_demand)
        
        # Execute capacity expansion
        if decision.capacity_expansion > 0:
            self.state.total_capacity += decision.capacity_expansion
            self.state.deployed_mass += decision.new_deployment
        
        # Calculate actual production (limited by capacity)
        max_production = self.state.total_capacity
        self.state.actual_production = min(decision.planned_production, max_production)
        self.state.production_history.append(self.state.actual_production)
        self.state.capacity_history.append(self.state.total_capacity)
        
        # Calculate earth supply requirements
        production_shortfall = max(0, self.state.current_demand - self.state.actual_production)
        self.state.earth_supply = min(decision.earth_supply_request, production_shortfall)
        
        # Update inventory
        inventory_change = (self.state.actual_production + self.state.earth_supply -
                          self.state.current_demand)
        self.state.inventory = max(0, self.state.inventory + inventory_change)
        
        # Update derived metrics
        self.state.update_derived_metrics()
        
        # Record decision
        self.decision_history.append(decision)
        
        return self.state
    
    def get_demand_trend(self, window: int = 3) -> Optional[List[float]]:
        """Get demand trend"""
        if len(self.state.demand_history) < window:
            return None
        return self.state.demand_history[-window:]
    
    def get_utilization_trend(self, window: int = 3) -> List[float]:
        """Get utilization trend"""
        if len(self.state.demand_history) < window or len(self.state.capacity_history) < window:
            return []
        
        utilizations = []
        for i in range(-window, 0):
            demand = self.state.demand_history[i]
            capacity = self.state.capacity_history[i]
            util = min(demand / max(capacity, 1e-6), 1.0)
            utilizations.append(util)
        
        return utilizations
    
    def calculate_performance_metrics(self, params: Dict) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            params: System parameters
            
        Returns:
            Performance metrics dictionary
        """
        if not self.decision_history:
            return {}
        
        # Basic statistics
        total_demand = sum(self.state.demand_history)
        total_production = sum(self.state.production_history)
        total_earth_supply = sum(d.earth_supply_request for d in self.decision_history)
        
        # Cost calculation
        total_expansion_cost = sum(d.expansion_cost for d in self.decision_history)
        total_operational_cost = sum(d.operational_cost for d in self.decision_history)
        total_supply_cost = sum(d.supply_cost for d in self.decision_history)
        total_cost = total_expansion_cost + total_operational_cost + total_supply_cost
        
        # Revenue calculation (assuming price per kg oxygen)
        oxygen_price = params.get('costs', {}).get('P_m', 21160)
        total_revenue = total_production * oxygen_price
        
        # NPV calculation (simplified version)
        discount_rate = params.get('economics', {}).get('r', 0.1)
        npv = 0
        for i, decision in enumerate(self.decision_history):
            year_revenue = self.state.production_history[i] * oxygen_price
            year_cost = decision.expansion_cost + decision.operational_cost + decision.supply_cost
            year_cash_flow = year_revenue - year_cost
            npv += year_cash_flow / ((1 + discount_rate) ** (i + 1))
        
        # Utilization statistics
        utilizations = []
        for i in range(len(self.state.demand_history)):
            if i < len(self.state.capacity_history):
                util = min(self.state.demand_history[i] / max(self.state.capacity_history[i], 1e-6), 1.0)
                utilizations.append(util)
        
        return {
            # Financial metrics
            'npv': npv,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_expansion_cost': total_expansion_cost,
            'total_operational_cost': total_operational_cost,
            'total_supply_cost': total_supply_cost,
            
            # Operational metrics
            'total_demand': total_demand,
            'total_production': total_production,
            'total_earth_supply': total_earth_supply,
            'self_sufficiency_rate': total_production / max(total_demand, 1e-6),
            'avg_utilization': np.mean(utilizations) if utilizations else 0,
            'min_utilization': min(utilizations) if utilizations else 0,
            'max_utilization': max(utilizations) if utilizations else 0,
            'utilization_std': np.std(utilizations) if utilizations else 0,
            
            # Capacity metrics
            'final_capacity': self.state.total_capacity,
            'final_deployed_mass': self.state.deployed_mass,
            'capacity_expansions': sum(1 for d in self.decision_history if d.capacity_expansion > 0),
            
            # Inventory metrics
            'final_inventory': self.state.inventory,
            'avg_inventory_days': np.mean([self.state.inventory_days]) if hasattr(self.state, 'inventory_days') else 0
        }
    
    def get_state_summary(self) -> Dict[str, float]:
        """Get current state summary"""
        return {
            'time': self.state.current_time,
            'total_capacity': self.state.total_capacity,
            'current_demand': self.state.current_demand,
            'utilization_rate': self.state.utilization_rate,
            'inventory': self.state.inventory,
            'self_sufficiency_rate': self.state.self_sufficiency_rate,
            'deployed_mass': self.state.deployed_mass
        }


if __name__ == "__main__":
    # Test code
    print("=== State Manager Test ===")
    
    # Create initial state
    initial_state = SystemState(
        total_capacity=100.0,
        deployed_mass=500.0,
        current_demand=80.0
    )
    
    manager = StateManager(initial_state)
    print(f"Initial state: {manager.get_state_summary()}")
    
    # Simulate a decision
    decision = Decision(
        time=1,
        capacity_expansion=20.0,
        new_deployment=100.0,
        planned_production=90.0,
        earth_supply_request=10.0,
        decision_reason="Demand growth, expand capacity"
    )
    
    # Update state
    params = {'economics': {'r': 0.1}, 'costs': {'P_m': 21160}}
    new_state = manager.update_state(decision, new_demand=110.0, params=params)
    
    print(f"Updated state: {manager.get_state_summary()}")
    
    # Calculate performance metrics
    metrics = manager.calculate_performance_metrics(params)
    print(f"Performance metrics: {metrics}")