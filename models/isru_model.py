"""
ISRU Oxygen Production Optimization Model
Complete optimization model integrating all model components
"""

import logging
from pyomo.environ import ConcreteModel, value as pyo_value
from typing import Dict, Any, Optional
import pandas as pd

from .variables import define_variables
from .constraints import define_constraints, validate_constraints
from .objective import define_objective, calculate_detailed_costs

logger = logging.getLogger(__name__)


class ISRUOptimizationModel:
    """
    ISRU Oxygen Production Optimization Model
    
    This is a complete Pyomo optimization model for optimizing the deployment and operation
    of lunar oxygen production systems. The model considers demand uncertainty, technical
    constraints, and cost factors.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize optimization model
        
        Args:
            params: Parameter dictionary containing all model parameters
        """
        self.params = params
        self.model = None
        self.demand_path = None
        self.solution = None
        
    def build_model(self, demand_path: list) -> ConcreteModel:
        """
        Build complete optimization model
        
        Args:
            demand_path: Demand path list
            
        Returns:
            Complete Pyomo model
        """
        self.demand_path = demand_path
        
        # Validate demand path length
        T = self.params['economics']['T']
        expected_length = T + 1  # Including time 0 to T
        
        if len(demand_path) != expected_length:
            raise ValueError(
                f"Demand path length error: expected {expected_length} (T={T} + 1), "
                f"actual {len(demand_path)}"
            )
        
        # Create model
        self.model = ConcreteModel(name="ISRU_Oxygen_Optimization")
        
        # Define variables
        logger.info("Defining decision variables...")
        self.model = define_variables(self.model, self.params)
        
        # Define constraints
        logger.info("Defining constraints...")
        self.model = define_constraints(self.model, self.params, demand_path)
        
        # Define objective function
        logger.info("Defining objective function...")
        self.model = define_objective(self.model, self.params)
        
        logger.info("Model construction completed")
        return self.model
    
    def solve(self, solver_name: str = 'glpk', solver_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve optimization model
        
        Args:
            solver_name: Solver name
            solver_options: Solver options
            
        Returns:
            Solution result dictionary
        """
        if self.model is None:
            raise ValueError("Model not built yet, please call build_model() first")
        
        try:
            # 导入求解器
            from pyomo.opt import SolverFactory
            
            # Create solver
            solver = SolverFactory(solver_name)
            
            if solver_options:
                for key, opt_val in solver_options.items():   # Avoid using 'value'
                    solver.options[key] = opt_val
            
            # Solve
            logger.info(f"Solving model using {solver_name} solver...")
            results = solver.solve(self.model, tee=False)
            
            # Check solution status
            if str(results.solver.termination_condition).lower() == 'optimal':
                logger.info("Model solved successfully")
                
                # Validate constraints
                if validate_constraints(self.model):
                    logger.info("All constraints validated")
                else:
                    logger.warning("Constraint violations detected")
                
                # Extract solution
                self.solution = self._extract_solution()
                
                # Calculate detailed costs
                self.solution['costs'] = calculate_detailed_costs(self.model, self.params)
                
                return {
                    'status': 'optimal',
                    'objective_value': pyo_value(self.model.NPV),
                    'solution': self.solution,
                    'solver_results': results
                }
            else:
                logger.error(f"Solution failed: {results.solver.termination_condition}")
                return {
                    'status': 'failed',
                    'termination_condition': str(results.solver.termination_condition),
                    'solver_results': results
                }
                
        except Exception as e:
            logger.error(f"Error during solving: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _extract_solution(self) -> Dict[str, Any]:
        """Extract solution results"""
        if self.model is None:
            return {}
        
        solution = {
            'Qt': {t: pyo_value(self.model.Qt[t]) for t in self.model.T},
            'Qt_cap': {t: pyo_value(self.model.Qt_cap[t]) for t in self.model.T},
            'St': {t: pyo_value(self.model.St[t]) for t in self.model.T},
            'Et': {t: pyo_value(self.model.Et[t]) for t in self.model.T},
            'Mt': {t: pyo_value(self.model.Mt[t]) for t in self.model.T},
            'delta_Mt': {t: pyo_value(self.model.delta_Mt[t]) for t in self.model.T},
            'M_leo': {t: pyo_value(self.model.M_leo[t]) for t in self.model.T},
            'NPV': pyo_value(self.model.NPV)
        }
        
        return solution
    
    def get_solution_dataframe(self) -> pd.DataFrame:
        """Convert solution to DataFrame format"""
        if self.solution is None:
            raise ValueError("Model not solved yet, please call solve() first")
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': list(self.solution['Qt'].keys()),
            'demand': self.demand_path,
            'Qt': list(self.solution['Qt'].values()),
            'Qt_cap': list(self.solution['Qt_cap'].values()),
            'St': list(self.solution['St'].values()),
            'Et': list(self.solution['Et'].values()),
            'Mt': list(self.solution['Mt'].values()),
            'delta_Mt': list(self.solution['delta_Mt'].values()),
            'M_leo': list(self.solution['M_leo'].values())
        })
        
        return df
    
    def save_solution(self, filepath: str) -> None:
        """Save solution results to file"""
        if self.solution is None:
            raise ValueError("Model not solved yet, please call solve() first")
        
        df = self.get_solution_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Solution results saved to: {filepath}")
    
    def print_solution_summary(self) -> None:
        """Print solution results summary"""
        if self.solution is None:
            print("Model not solved yet")
            return
        
        T = self.params['economics']['T']
        
        print(f"\n{'='*60}")
        print(f"ISRU Optimization Results (T={T} years)")
        print(f"{'='*60}")
        
        # Core metrics
        print(f"\nCore Metrics")
        print(f"{'─'*40}")
        print(f"Net Present Value : ${self.solution['NPV']:>15,.2f}")
        print(f"Solution Status   : {'Optimal' if self.solution else 'Failed'}")
        
        # Input parameters (key assumptions)
        print(f"\nKey Input Parameters")
        print(f"{'─'*40}")
        econ = self.params['economics']
        demand = self.params['demand']
        print(f"Time Horizon      : {T:>15} years")
        print(f"Discount Rate     : {econ['r']*100:>14.1f}%")
        print(f"Initial Demand    : {demand['D0']:>11,.0f} kg/year")
        print(f"Demand Growth (μ) : {demand['mu']*100:>14.1f}%")
        print(f"Demand Volatility : {demand['sigma']*100:>14.1f}%")
        
        # Decision variables summary
        print(f"\nDecision Variables Summary")
        print(f"{'─'*40}")
        total_delivery = sum(self.solution['Qt'].values())
        total_shortage = sum(self.solution['St'].values())
        total_excess = sum(self.solution['Et'].values())
        max_isru_mass = max(self.solution['Mt'].values())
        total_deployment = sum(self.solution['delta_Mt'].values())
        total_leo_mass = sum(self.solution['M_leo'].values())
        
        print(f"Total Oxygen Delivery : {total_delivery:>11,.0f} kg")
        print(f"Total Shortage        : {total_shortage:>11,.0f} kg")
        print(f"Total Excess          : {total_excess:>11,.0f} kg")
        print(f"Max ISRU Mass         : {max_isru_mass:>11,.0f} kg")
        print(f"Total New Deployment  : {total_deployment:>11,.0f} kg")
        print(f"Total LEO Transport   : {total_leo_mass:>11,.0f} kg")
        
        # Economic analysis
        if 'costs' in self.solution:
            from .objective import print_cost_breakdown
            print_cost_breakdown(self.solution['costs'])
        
        print(f"\n{'='*60}")


# Convenience function
def create_and_solve_model(params: Dict[str, Any], demand_path: list,
                          solver_name: str = 'glpk') -> Dict[str, Any]:
    """
    Convenience function: create and solve model
    
    Args:
        params: Parameter dictionary
        demand_path: Demand path
        solver_name: Solver name
        
    Returns:
        Solution results
    """
    model = ISRUOptimizationModel(params)
    model.build_model(demand_path)
    return model.solve(solver_name)


if __name__ == "__main__":
    # Test code
    import json
    
    # Load parameters
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    
    # Create demand path (using average demand)
    T = params['economics']['T']
    D0 = params['demand']['D0']
    demand_path = [0] + [D0 * (1.02 ** t) for t in range(1, T + 1)]
    
    # Create and solve model
    result = create_and_solve_model(params, demand_path)
    
    if result['status'] == 'optimal':
        print("Model solved successfully!")
        print(f"Optimal objective value: ${result['objective_value']:,.2f}")
    else:
        print(f"Solution failed: {result['status']}")
