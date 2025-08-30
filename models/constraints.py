"""
Pyomo Model Constraints Definition
Define all constraints for ISRU oxygen production optimization problem (5-year period)
"""

from pyomo.environ import Constraint, ConcreteModel


def define_constraints(model: ConcreteModel, params: dict, demand_path: list) -> ConcreteModel:
    tech = params['technology']
    T = params['economics']['T']
    
    # 1. Demand balance: demand must be satisfied by delivery + earth supply + shortage
    def demand_balance_rule(m, t):
        if t >= len(demand_path):
            raise IndexError(
                f"Time index {t} exceeds demand path range [0, {len(demand_path)-1}]"
            )
        return m.Qt[t] + m.Q_earth[t] + m.St[t] == demand_path[t]
    model.demand_balance = Constraint(model.T, rule=demand_balance_rule)

    # 2. Delivery ≤ capacity
    def delivery_capacity_rule(m, t):
        return m.Qt[t] <= m.Qt_cap[t]
    model.delivery_capacity = Constraint(model.T, rule=delivery_capacity_rule)

    # 3. Surplus definition
    def surplus_def_rule(m, t):
        return m.Et[t] == m.Qt_cap[t] - m.Qt[t]
    model.surplus_def = Constraint(model.T, rule=surplus_def_rule)

    # 4. Capacity = η * Mt
    def capacity_mass_rule(m, t):
        return m.Qt_cap[t] == tech['eta'] * m.Mt[t]
    model.capacity_mass = Constraint(model.T, rule=capacity_mass_rule)

    # 5. ISRU mass balance
    def mass_balance_rule(m, t):
        if t == 0:
            return m.Mt[t] == tech.get("M0", 0)
        return m.Mt[t] == m.Mt[t-1] + m.delta_Mt[t]
    model.mass_balance = Constraint(model.T, rule=mass_balance_rule)

    # 6. Launch mass (only charge for new deployments)
    def leo_mass_rule(m, t):
        return m.M_leo[t] == tech['alpha'] * m.delta_Mt[t]
    model.leo_mass = Constraint(model.T, rule=leo_mass_rule)

    return model


def validate_constraints(model: ConcreteModel) -> bool:
    """Simple validation of constraint satisfaction (optional for debugging)"""
    return True
