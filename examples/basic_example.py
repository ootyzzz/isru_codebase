#!/usr/bin/env python3
"""
ISRU Model Basic Usage Example
Demonstrates how to load parameters, generate demand paths, and solve the model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from models.isru_model import create_and_solve_model
from analysis.gbm_demand import GBMDemandGenerator

def main():
    """Run basic example"""
    print("=== ISRU Model Basic Example ===\n")
    
    # 1. Load parameters
    print("1. Loading parameters...")
    with open("data/parameters.json") as f:
        params = json.load(f)
    
    # Print key parameters
    T = params['economics']['T']
    print(f"   Time period: {T} months")
    print(f"   Oxygen price: ${params['economics']['P_m']:,}")
    print(f"   Initial demand: {params['demand']['D0']} kg")
    
    # 2. Create demand generator
    print("\n2. Creating GBM demand generator...")
    demand_params = params['demand']
    gbm_generator = GBMDemandGenerator(
        D0=demand_params['D0'],
        mu=demand_params['mu'],
        sigma=demand_params['sigma'],
        dt=demand_params['dt']
    )
    
    # 3. Generate demand path
    print("\n3. Generating demand path...")
    np.random.seed(42)  # Ensure reproducible results
    demand_path = gbm_generator.generate_single_path(T)
    
    print(f"   Demand path length: {len(demand_path)}")
    print(f"   Initial demand: {demand_path[0]:.2f} kg")
    print(f"   Final demand: {demand_path[-1]:.2f} kg")
    print(f"   Average demand: {np.mean(demand_path):.2f} kg")
    
    # 4. Solve model
    print("\n4. Solving optimization model...")
    result = create_and_solve_model(params, demand_path, solver_name="glpk")
    
    if result['status'] == 'optimal':
        print("   ✅ Solving successful!")
        print(f"   NPV: ${result['objective_value']:,.2f}")
        
        # Extract key decision variables
        solution = result['solution']
        Mt_total = sum(solution['delta_Mt'].values())
        print(f"   Total extraction: {Mt_total:.2f} kg")
        
        # Print period-by-period decisions
        print("\n5. Period-by-period decisions:")
        for t in range(T):
            print(f"   Month {t+1:2d}: "
                  f"Extract {solution['delta_Mt'][t]:6.2f} kg, "
                  f"Inventory {solution['St'][t]:6.2f} kg, "
                  f"Demand {demand_path[t+1]:6.2f} kg")
        
        # Save results
        with open('data/example_results.json', 'w') as f:
            json.dump({
                'parameters': params,
                'demand_path': demand_path.tolist(),
                'solution': solution,
                'npv': result['objective_value']
            }, f, indent=2)
        print("\n6. Results saved to data/example_results.json")
        
    else:
        print(f"   ❌ Solving failed: {result['status']}")

if __name__ == "__main__":
    main()