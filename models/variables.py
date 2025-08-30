"""
Pyomo Model Decision Variables Definition
Define all decision variables for ISRU oxygen production optimization problem
"""

from pyomo.environ import ConcreteModel, Var, NonNegativeReals, RangeSet


def define_variables(model: ConcreteModel, params: dict) -> ConcreteModel:
    """
    Define decision variables for optimization model
    
    Args:
        model: Pyomo model instance
        params: Parameter dictionary
        
    Returns:
        Pyomo model containing variables
    """
    T = params['economics']['T']
    
    # Time index
    model.T = RangeSet(0, T)
    
    # Decision variables
    # Qt: Delivered oxygen quantity [kg]
    model.Qt = Var(model.T, within=NonNegativeReals, doc="Delivered oxygen quantity")
    
    # Qt_cap: Production capacity [kg]
    model.Qt_cap = Var(model.T, within=NonNegativeReals, doc="Production capacity")
    
    # St: Shortage quantity [kg]
    model.St = Var(model.T, within=NonNegativeReals, doc="Shortage quantity")
    
    # Et: Excess quantity [kg]
    model.Et = Var(model.T, within=NonNegativeReals, doc="Excess quantity")
    
    # Mt: Deployed ISRU mass [kg]
    model.Mt = Var(model.T, within=NonNegativeReals, doc="Deployed ISRU mass")
    
    # delta_Mt: New ISRU deployment [kg]
    model.delta_Mt = Var(model.T, within=NonNegativeReals, doc="New ISRU deployment")
    
    # M_leo: Mass launched to LEO [kg]
    model.M_leo = Var(model.T, within=NonNegativeReals, doc="Mass launched at LEO")

    # Q_earth: Earth-supplied oxygen quantity [kg]
    model.Q_earth = Var(model.T, within=NonNegativeReals, doc="Earth-supplied oxygen quantity")

    return model


def get_variable_names() -> dict:
    """Return name mapping for all decision variables"""
    return {
        'Qt': 'Delivered oxygen quantity',
        'Qt_cap': 'Production capacity',
        'St': 'Shortage quantity',
        'Et': 'Excess quantity',
        'Mt': 'Deployed ISRU mass',
        'delta_Mt': 'New ISRU deployment',
        'M_leo': 'Mass launched at LEO',
        'Q_earth': 'Earth-supplied oxygen quantity'
    }


def get_variable_bounds(params: dict) -> dict:
    """Return reasonable bounds for variables"""
    max_demand = params['demand']['D0'] * 5  # Maximum demand estimate
    
    return {
        'Qt': (0, max_demand),
        'Qt_cap': (0, max_demand * 2),
        'St': (0, max_demand),
        'Et': (0, max_demand),
        'Mt': (0, max_demand * 1000),        # Consider mass conversion
        'delta_Mt': (0, max_demand * 1000),
        'M_leo': (0, max_demand * 1000),      # Same order as Mt
        'Q_earth': (0, max_demand * 2)
    }
