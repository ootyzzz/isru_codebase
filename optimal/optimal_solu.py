"""
Test ISRU Optimization Model (t-year period, GBM stochastic demand)
"""

import json
import numpy as np
import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.isru_model import create_and_solve_model
from analysis.gbm_demand import GBMDemandGenerator

def solve_isru_optimization(T_years, params_file="data/parameters.json", random_seed=42, verbose=True):
    """
    Core function to solve ISRU optimization problem
    
    Args:
        T_years: Time length (years)
        params_file: Parameter file path
        random_seed: Random seed
        verbose: Whether to print detailed information
        
    Returns:
        dict: Dictionary containing solving results
    """
    # Load parameters
    with open(params_file, "r") as f:
        params = json.load(f)
    
    # Modify time parameter
    params["economics"]["T"] = T_years
    
    # Set random seed to ensure reproducible results
    np.random.seed(random_seed)
    
    # Use GBM to generate stochastic demand path
    expected_length = T_years + 1
    demand_params = params['demand']
    gbm_generator = GBMDemandGenerator(
        D0=demand_params['D0'],
        mu=demand_params['mu'],
        sigma=demand_params['sigma'],
        dt=demand_params['dt']
    )
    demand_path = gbm_generator.generate_single_path(expected_length)
    
    # Validate demand path length - GBM generator returns T+1 points (including t=0)
    if len(demand_path) != expected_length:
        # Adjust demand path, remove initial point at t=0
        demand_path = demand_path[1:]  # Keep only t=1 to t=T
        if verbose:
            print(f"Adjusted demand path length: {len(demand_path)} (expected: {T_years})")
    
    # Ensure demand_path is a list rather than numpy array
    if hasattr(demand_path, 'tolist'):
        demand_path = demand_path.tolist()

    # Run optimization
    result = create_and_solve_model(params, demand_path, solver_name="glpk")
    
    # Construct return result
    output = {
        "T": T_years,
        "status": result["status"],
        "npv": result.get("objective_value", None),
        "solve_time": result.get("solve_time", None),
        "demand_path": demand_path,
        "params": params,
        "full_result": result
    }
    
    # Print results (if needed)
    if verbose:
        if result["status"] == "optimal":
            print(f"T={T_years}: NPV = {result['objective_value']:,.2f}")
        else:
            print(f"T={T_years}: Solving failed - {result['status']}")
            if "termination_condition" in result:
                print(f"   Reason: {result['termination_condition']}")
    
    return output

def main():
    """Run single optimization (maintain original functionality)"""
    # Load parameters to get default T value
    with open("data/parameters.json", "r") as f:
        params = json.load(f)
    
    T = params["economics"]["T"]
    result = solve_isru_optimization(T, verbose=True)
    
    # Print detailed result summary
    if result["status"] == "optimal":
        print("\n=== Detailed Result Summary ===")
        from models.isru_model import ISRUOptimizationModel
        model = ISRUOptimizationModel(result["params"])
        model.solution = result["full_result"]["solution"]
        model.demand_path = result["demand_path"]
        model.print_solution_summary()
    else:
        if "termination_condition" in result["full_result"]:
            print(f"Reason: {result['full_result']['termination_condition']}")
        if "error" in result["full_result"]:
            print(f"Error: {result['full_result']['error']}")

if __name__ == "__main__":
    main()
