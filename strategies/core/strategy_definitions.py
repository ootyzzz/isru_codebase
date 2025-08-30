#!/usr/bin/env python3
"""
ISRU Strategy Definition Module - New Strategy Version
Define parameters and decision logic for three new deployment strategies
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class StrategyType(Enum):
    """Strategy type enumeration"""
    UPFRONT_DEPLOYMENT = "upfront_deployment"
    GRADUAL_DEPLOYMENT = "gradual_deployment"
    FLEXIBLE_DEPLOYMENT = "flexible_deployment"


@dataclass
class StrategyParams:
    """Strategy parameters data structure"""
    name: str
    description: str
    
    # Deployment strategy type
    deployment_type: str             # "upfront", "gradual", "flexible"
    
    # Time horizon related parameters
    time_horizon: int                # Total time horizon T
    
    # Deployment calculation parameters
    total_deployment_ratio: float    # Total deployment ratio relative to final demand
    
    # Flexible strategy specific parameters
    response_threshold: float        # Response threshold (supply-demand difference ratio)
    
    def __post_init__(self):
        """Validate parameter validity"""
        assert self.deployment_type in ["upfront", "gradual", "flexible"], "Deployment type must be upfront, gradual, or flexible"
        assert self.time_horizon > 0, "Time horizon must be greater than 0"
        assert 0 <= self.total_deployment_ratio <= 2.0, "Total deployment ratio should be between 0-2.0"
        assert 0 <= self.response_threshold <= 1.0, "Response threshold should be between 0-1.0"


class StrategyDefinitions:
    """Strategy definition class - New strategy version"""
    
    @staticmethod
    def get_upfront_deployment_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        Upfront deployment strategy: Deploy total final deployment in first year
        - Large upfront investment, deploy all at once
        - No new deployments in subsequent years, only maintain operations
        - Determine total deployment based on T-year demand expectations
        """
        return StrategyParams(
            name="upfront_deployment",
            description="Upfront Deployment: Deploy all capacity in the first year",
            deployment_type="upfront",
            time_horizon=time_horizon,
            total_deployment_ratio=1.0,     # Deploy 100% of final demand
            response_threshold=0.0          # Not applicable
        )
    
    @staticmethod
    def get_gradual_deployment_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        Gradual deployment strategy: Distribute total deployment evenly across all years
        - Evenly distributed investment, steady expansion
        - Annual increment = Final total deployment / T
        - System knows time horizon T from the beginning
        """
        return StrategyParams(
            name="gradual_deployment",
            description="Gradual Deployment: Distribute total capacity evenly across all years",
            deployment_type="gradual",
            time_horizon=time_horizon,
            total_deployment_ratio=1.0,     # Deploy 100% of final demand
            response_threshold=0.0          # Not applicable
        )
    
    @staticmethod
    def get_flexible_deployment_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        Flexible deployment strategy: Dynamically adjust based on actual supply-demand conditions
        - Responsive deployment, dynamically adjust based on supply-demand differences
        - If year t supply < demand, then deploy the gap in year t+1
        - Reactive strategy, always trying to fill the previous year's gap
        """
        return StrategyParams(
            name="flexible_deployment",
            description="Flexible Deployment: Respond to supply-demand gaps dynamically",
            deployment_type="flexible",
            time_horizon=time_horizon,
            total_deployment_ratio=1.0,     # Theoretical maximum deployment ratio
            response_threshold=0.05         # 5% response threshold
        )
    
    @classmethod
    def get_strategy(cls, strategy_type: StrategyType, time_horizon: int = 20) -> StrategyParams:
        """Get strategy parameters based on strategy type"""
        strategy_map = {
            StrategyType.UPFRONT_DEPLOYMENT: lambda: cls.get_upfront_deployment_strategy(time_horizon),
            StrategyType.GRADUAL_DEPLOYMENT: lambda: cls.get_gradual_deployment_strategy(time_horizon),
            StrategyType.FLEXIBLE_DEPLOYMENT: lambda: cls.get_flexible_deployment_strategy(time_horizon)
        }
        
        if strategy_type not in strategy_map:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_map[strategy_type]()
    
    @classmethod
    def get_all_strategies(cls, time_horizon: int = 20) -> Dict[str, StrategyParams]:
        """Get all strategies"""
        return {
            "upfront_deployment": cls.get_upfront_deployment_strategy(time_horizon),
            "gradual_deployment": cls.get_gradual_deployment_strategy(time_horizon),
            "flexible_deployment": cls.get_flexible_deployment_strategy(time_horizon)
        }


def calculate_deployment_for_year(strategy: StrategyParams,
                                year: int,
                                final_demand: float,
                                previous_supply: float = 0,
                                previous_demand: float = 0) -> float:
    """
    Calculate deployment amount for a given year based on strategy
    
    Args:
        strategy: Strategy parameters
        year: Current year (0-based)
        final_demand: Expected demand in final year
        previous_supply: Total supply from previous year (used only by flexible strategy)
        previous_demand: Demand from previous year (used only by flexible strategy)
        
    Returns:
        New deployment amount for current year
    """
    total_capacity_needed = final_demand * strategy.total_deployment_ratio
    
    if strategy.deployment_type == "upfront":
        # Upfront deployment: Deploy all in first year, zero in other years
        return total_capacity_needed if year == 0 else 0
        
    elif strategy.deployment_type == "gradual":
        # Gradual deployment: Deploy same amount each year
        return total_capacity_needed / strategy.time_horizon
        
    elif strategy.deployment_type == "flexible":
        # Flexible deployment: Decide based on previous year's supply-demand gap
        if year == 0:
            # Deploy a base amount in first year (e.g., 20% of final demand)
            return total_capacity_needed * 0.2
        else:
            # Decide based on previous year's supply-demand gap
            if previous_demand > previous_supply:
                gap = previous_demand - previous_supply
                gap_ratio = gap / previous_demand if previous_demand > 0 else 0
                # If gap exceeds threshold, deploy the gap amount
                if gap_ratio > strategy.response_threshold:
                    return gap
            return 0
    
    return 0


def should_expand_capacity(strategy: StrategyParams,
                          current_utilization: float,
                          current_capacity: float,
                          demand_trend: Optional[List[float]] = None) -> bool:
    """
    Determine whether capacity should be expanded - New strategy version
    
    Args:
        strategy: Strategy parameters
        current_utilization: Current utilization rate
        current_capacity: Current capacity
        demand_trend: Demand trend (optional)
        
    Returns:
        Whether expansion should occur
    """
    # New strategies don't use traditional utilization-triggered expansion
    # Expansion decisions are entirely determined by strategy deployment logic
    return False


def calculate_expansion_amount(strategy: StrategyParams,
                             current_capacity: float,
                             current_demand: float) -> float:
    """
    Calculate expansion amount - New strategy version
    
    Args:
        strategy: Strategy parameters
        current_capacity: Current capacity
        current_demand: Current demand
        
    Returns:
        Expansion amount
    """
    # New strategy expansion amounts are determined by deployment logic, return 0 here
    return 0


if __name__ == "__main__":
    # Test code
    print("=== ISRU New Strategy Definition Test ===")
    
    time_horizon = 20
    strategies = StrategyDefinitions.get_all_strategies(time_horizon)
    
    for name, strategy in strategies.items():
        print(f"\n{name.upper()} Strategy:")
        print(f"  Description: {strategy.description}")
        print(f"  Deployment Type: {strategy.deployment_type}")
        print(f"  Time Horizon: {strategy.time_horizon}")
        print(f"  Total Deployment Ratio: {strategy.total_deployment_ratio:.1%}")
        print(f"  Response Threshold: {strategy.response_threshold:.1%}")
        
        # Test deployment logic
        final_demand = 1000
        print(f"  Deployment Test (Final Demand: {final_demand} kg):")
        
        for year in range(min(5, time_horizon)):
            deployment = calculate_deployment_for_year(
                strategy, year, final_demand,
                previous_supply=800 if year > 0 else 0,
                previous_demand=900 if year > 0 else 0
            )
            print(f"    Year {year+1} Deployment: {deployment:.1f} kg")