"""
GLPK Solver Interface
Provides integration interface with GLPK solver
"""

import logging
import os
import subprocess
from typing import Dict, Any, Optional
from pyomo.environ import SolverFactory
from pyomo.opt import SolverResults

logger = logging.getLogger(__name__)


class GLPKSolver:
    """
    GLPK solver wrapper class
    
    Provides complete interface to GLPK solver, including solver configuration,
    option settings, and result processing.
    """
    
    def __init__(self, solver_path: Optional[str] = None):
        """
        Initialize GLPK solver
        
        Args:
            solver_path: GLPK solver executable path (optional)
        """
        self.solver_path = solver_path
        self.solver = None
        self._setup_solver()
    
    def _setup_solver(self) -> None:
        """Setup GLPK solver"""
        try:
            # Try to create solver instance
            self.solver = SolverFactory('glpk')
            
            if self.solver is None or not self.solver.available():
                raise RuntimeError("GLPK solver not available")
            
            logger.info("GLPK solver initialized successfully")
            
        except Exception as e:
            logger.error(f"GLPK solver initialization failed: {e}")
            # Try to install GLPK
            self._install_glpk()
    
    def _install_glpk(self) -> None:
        """Try to install GLPK solver"""
        logger.info("Attempting to install GLPK solver...")
        
        try:
            # Check if conda is available
            result = subprocess.run(['conda', '--version'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Installing glpk using conda...")
                subprocess.run(['conda', 'install', '-y', 'glpk'], check=True)
                logger.info("GLPK installation successful")
                self.solver = SolverFactory('glpk')
            else:
                logger.warning("conda not available, please install GLPK manually")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"GLPK installation failed: {e}")
            logger.info("Please install GLPK manually: conda install glpk")
    
    def solve(self, model, **kwargs) -> Dict[str, Any]:
        """
        Solve optimization model
        
        Args:
            model: Pyomo model
            **kwargs: Solver options
            
        Returns:
            Solving result dictionary
        """
        if self.solver is None:
            raise RuntimeError("GLPK solver not initialized")
        
        # Default solver options
        default_options = {
            'tmlim': 3600,  # Time limit (seconds)
            'mipgap': 0.01,  # MIP gap
            'msg_lev': 3,   # Message level
            'presolve': True,
            'scale': True
        }
        
        # Merge user options
        solver_options = {**default_options, **kwargs}
        
        try:
            # Set solver options
            for key, value in solver_options.items():
                self.solver.options[key] = value
            
            # Solve model
            logger.info("Starting model solving...")
            results = self.solver.solve(model, tee=True)
            
            # Process results
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"Error during solving: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _process_results(self, results: SolverResults) -> Dict[str, Any]:
        """Process solving results"""
        status_map = {
            'optimal': 'optimal',
            'infeasible': 'infeasible',
            'unbounded': 'unbounded',
            'limit': 'limit',
            'error': 'error',
            'unknown': 'unknown'
        }
        
        solver_status = str(results.solver.termination_condition)
        status = status_map.get(solver_status, 'unknown')
        
        result = {
            'status': status,
            'solver_status': solver_status,
            'solve_time': results.solver.time,
            'iterations': results.solver.iterations,
            'objective_value': None
        }
        
        if status == 'optimal':
            # Extract objective value
            from pyomo.environ import value
            if hasattr(results.problem, 'objective'):
                result['objective_value'] = value(results.problem.objective)
        
        return result
    
    def get_solver_info(self) -> Dict[str, Any]:
        """Get solver information"""
        if self.solver is None:
            return {'available': False}
        
        return {
            'available': True,
            'name': 'GLPK',
            'version': self.solver.version(),
            'executable': self.solver.executable()
        }


class SolverConfig:
    """Solver configuration class"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default solver configuration"""
        return {
            'solver_name': 'glpk',
            'options': {
                'tmlim': 3600,
                'mipgap': 0.01,
                'msg_lev': 3,
                'presolve': True,
                'scale': True
            }
        }
    
    @staticmethod
    def get_fast_config() -> Dict[str, Any]:
        """Get fast solving configuration"""
        return {
            'solver_name': 'glpk',
            'options': {
                'tmlim': 300,
                'mipgap': 0.05,
                'msg_lev': 2,
                'presolve': True,
                'scale': True
            }
        }
    
    @staticmethod
    def get_precise_config() -> Dict[str, Any]:
        """Get precise solving configuration"""
        return {
            'solver_name': 'glpk',
            'options': {
                'tmlim': 7200,
                'mipgap': 0.001,
                'msg_lev': 4,
                'presolve': True,
                'scale': True,
                'cuts': True
            }
        }


def check_glpk_availability() -> bool:
    """Check if GLPK is available"""
    try:
        solver = SolverFactory('glpk')
        return solver.available()
    except:
        return False


def get_glpk_installation_guide() -> str:
    """Get GLPK installation guide"""
    return """
    GLPK Installation Guide:
    
    1. Install using conda (recommended):
       conda install glpk
    
    2. Install using pip:
       pip install glpk
    
    3. Manual installation:
       - Windows: Download GLPK for Windows
       - Linux: sudo apt-get install glpk-utils
       - macOS: brew install glpk
    
    4. Verify installation:
       python -c "from pyomo.environ import SolverFactory; print(SolverFactory('glpk').available())"
    """


if __name__ == "__main__":
    # Test GLPK solver
    print("Checking GLPK availability...")
    
    if check_glpk_availability():
        print("✓ GLPK solver available")
        
        solver = GLPKSolver()
        info = solver.get_solver_info()
        print(f"Solver information: {info}")
    else:
        print("✗ GLPK solver not available")
        print(get_glpk_installation_guide())