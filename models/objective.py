"""
Pyomo Model Objective Function Definition
Define objective function for ISRU oxygen production optimization problem
"""

from pyomo.environ import Objective, quicksum, value, ConcreteModel, maximize


def define_objective(model: ConcreteModel, params: dict) -> ConcreteModel:
    econ = params['economics']
    tech = params['technology']
    costs = params['costs']   # Key: added here
    
    discount_factor = [(1 + econ['r']) ** (-t) for t in model.T]

    # Revenue
    revenue = quicksum(
        econ['P_m'] * model.Qt[t] * discount_factor[t]
        for t in model.T
    )
    
    # Cost components
    launch_cost = quicksum(
        costs['c_L'] * model.M_leo[t] * discount_factor[t]
        for t in model.T
    )
    development_cost = quicksum(
        costs['c_dev'] * model.delta_Mt[t] * discount_factor[t]
        for t in model.T
    )
    operating_cost = quicksum(
        costs['c_op'] * model.Mt[t] * discount_factor[t]
        for t in model.T
    )
    shortage_cost = quicksum(
        costs['c_bu'] * model.St[t] * discount_factor[t]
        for t in model.T
    )
    storage_cost = quicksum(
        costs['c_S'] * model.Et[t] * discount_factor[t]
        for t in model.T
    )
    byproduct_cost = quicksum(
        costs['c_by'] * tech['beta'] * model.Qt[t] * discount_factor[t]
        for t in model.T
    )
    earth_supply_cost = quicksum(
        costs['c_E'] * model.Q_earth[t] * discount_factor[t]
        for t in model.T
    )

    total_cost = (
        launch_cost + development_cost + operating_cost +
        shortage_cost + storage_cost + byproduct_cost + earth_supply_cost
    )
    
    model.NPV = revenue - total_cost
    model.objective = Objective(expr=model.NPV, sense=maximize)
    
    return model

def calculate_detailed_costs(model: ConcreteModel, params: dict) -> dict:
    """Calculate detailed cost breakdown"""
    econ = params['economics']
    tech = params['technology']
    costs = params['costs']

    discount_factor = [(1 + econ['r']) ** (-t) for t in model.T]

    detailed_costs = {
        'revenue': sum(
            econ['P_m'] * value(model.Qt[t]) * discount_factor[t] 
            for t in model.T
        ),
        'launch_cost': sum(
            costs['c_L'] * value(model.M_leo[t]) * discount_factor[t]
            for t in model.T
        ),
        'development_cost': sum(
            costs['c_dev'] * value(model.delta_Mt[t]) * discount_factor[t]
            for t in model.T
        ),
        'operating_cost': sum(
            costs['c_op'] * value(model.Mt[t]) * discount_factor[t]
            for t in model.T
        ),
        'shortage_cost': sum(
            costs['c_bu'] * value(model.St[t]) * discount_factor[t]
            for t in model.T
        ),
        'storage_cost': sum(
            costs['c_S'] * value(model.Et[t]) * discount_factor[t]
            for t in model.T
        ),
        'byproduct_cost': sum(
            costs['c_by'] * tech['beta'] * value(model.Qt[t]) * discount_factor[t]
            for t in model.T
        ),
        'earth_supply_cost': sum(
            costs['c_E'] * value(model.Q_earth[t]) * discount_factor[t]
            for t in model.T
        )

    }

    detailed_costs['total_cost'] = (
        detailed_costs['launch_cost'] + detailed_costs['development_cost'] +
        detailed_costs['operating_cost'] + detailed_costs['shortage_cost'] +
        detailed_costs['storage_cost'] + detailed_costs['byproduct_cost'] + detailed_costs['earth_supply_cost']
    )
    detailed_costs['NPV'] = detailed_costs['revenue'] - detailed_costs['total_cost']
    
    return detailed_costs

def print_cost_breakdown(costs: dict) -> None:
    """Print cost breakdown"""
    print(f"\nEconomic Analysis")
    print(f"{'─'*40}")
    
    # Revenue section
    print(f"Revenue")
    print(f"  Total Revenue     : ${costs['revenue']:>15,.2f}")
    
    # Cost section - grouped by logic
    print(f"\nCost Components")
    print(f"  Development Cost  : ${costs['development_cost']:>15,.2f}")
    print(f"  Operating Cost    : ${costs['operating_cost']:>15,.2f}")
    print(f"  Launch Cost       : ${costs['launch_cost']:>15,.2f}")
    print(f"  Storage Cost      : ${costs['storage_cost']:>15,.2f}")
    print(f"  Byproduct Cost    : ${costs['byproduct_cost']:>15,.2f}")
    print(f"  Earth Supply Cost : ${costs['earth_supply_cost']:>15,.2f}")
    
    # Penalty costs (if any)
    shortage_cost = costs.get('shortage_cost', 0)
    if shortage_cost > 0:
        print(f"  Shortage Penalty  : ${shortage_cost:>15,.2f}")
    
    build_cost = costs.get('build_cost', 0)
    if build_cost > 0:
        print(f"  Construction Cost : ${build_cost:>15,.2f}")
    
    # Summary
    print(f"  {'─'*32}")
    print(f"  Total Cost        : ${costs['total_cost']:>15,.2f}")
    
    # Net present value
    print(f"\nFinal Result")
    print(f"  Net Present Value : ${costs['NPV']:>15,.2f}")

