"""
Geometric Brownian Motion (GBM) Demand Generator
For generating Monte Carlo simulation paths of lunar oxygen demand
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GBMDemandGenerator:
    """
    Geometric Brownian Motion demand generator
    
    Implements formula: D_t = D_{t-1} * exp((μ - 0.5σ²)dt + σ√dt * Z_t)
    where Z_t ~ N(0,1) is a standard normal random variable
    """
    
    def __init__(self, D0: float, mu: float, sigma: float, dt: float = 1.0):
        """
        Initialize GBM generator
        
        Args:
            D0: Initial demand
            mu: Drift rate
            sigma: Volatility
            dt: Time step
        """
        self.D0 = D0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        
    def generate_single_path(self, T: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate single demand path
        
        Args:
            T: Time range (years)
            seed: Random seed
            
        Returns:
            Demand path array with shape (T+1,)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize path
        path = np.zeros(T + 1)
        path[0] = self.D0
        
        # GBM parameters
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        # Generate path
        for t in range(1, T + 1):
            z = np.random.standard_normal()
            path[t] = path[t-1] * np.exp(drift + diffusion * z)
            
        return path
    
    def generate_multiple_paths(self, T: int, n_scenarios: int, 
                               seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate multiple demand paths
        
        Args:
            T: Time range (years)
            n_scenarios: Number of scenarios
            seed: Random seed
            
        Returns:
            DataFrame where each column is a demand path
        """
        if seed is not None:
            np.random.seed(seed)
            
        paths = []
        for scenario in range(n_scenarios):
            path = self.generate_single_path(T)
            paths.append(path)
            
        # Create DataFrame
        df = pd.DataFrame(
            np.array(paths).T,
            columns=[f'scenario_{i+1}' for i in range(n_scenarios)],
            index=range(T + 1)
        )
        
        return df
    
    def generate_scenarios_with_stats(self, T: int, n_scenarios: int, 
                                    seed: Optional[int] = None) -> dict:
        """
        Generate demand scenarios and calculate statistics
        
        Returns:
            Dictionary containing paths and statistics
        """
        df = self.generate_multiple_paths(T, n_scenarios, seed)
        
        stats = {
            'paths': df,
            'mean': df.mean(axis=1),
            'std': df.std(axis=1),
            'percentile_5': df.quantile(0.05, axis=1),
            'percentile_95': df.quantile(0.95, axis=1),
            'min': df.min(axis=1),
            'max': df.max(axis=1)
        }
        
        return stats
    
    def save_scenarios(self, df: pd.DataFrame, filepath: str) -> None:
        """Save demand scenarios to CSV file"""
        df.to_csv(filepath)
        logger.info(f"Demand scenarios saved to: {filepath}")
        
    def load_scenarios(self, filepath: str) -> pd.DataFrame:
        """Load demand scenarios from CSV file"""
        df = pd.read_csv(filepath, index_col=0)
        logger.info(f"Demand scenarios loaded from {filepath}")
        return df


def create_demand_generator_from_params(params: dict) -> GBMDemandGenerator:
    """
    Create GBM generator from parameter dictionary
    
    Args:
        params: Parameter dictionary containing demand parameters
        
    Returns:
        GBMDemandGenerator instance
    """
    demand_params = params['demand']
    return GBMDemandGenerator(
        D0=demand_params['D0'],
        mu=demand_params['mu'],
        sigma=demand_params['sigma'],
        dt=demand_params['dt']
    )


if __name__ == "__main__":
    # Test code
    import json
    
    # Load parameters
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    
    # Create generator
    generator = create_demand_generator_from_params(params)
    
    # Generate scenarios
    T = params['economics']['T']
    n_scenarios = params['simulation']['n_scenarios']
    seed = params['simulation']['random_seed']
    
    stats = generator.generate_scenarios_with_stats(T, n_scenarios, seed)
    
    # Save results
    generator.save_scenarios(stats['paths'], 'data/demand_scenarios.csv')
    
    # Print statistics
    print("Demand scenario statistics:")
    print(stats['mean'])
    print(f"\nNumber of scenarios: {n_scenarios}")
    print(f"Time range: {T} years")