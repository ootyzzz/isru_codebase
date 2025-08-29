"""
Pyomo模型目标函数定义
定义ISRU氧气生产优化问题的目标函数
"""

from pyomo.environ import Objective, quicksum, value, ConcreteModel, maximize


def define_objective(model: ConcreteModel, params: dict) -> ConcreteModel:
    econ = params['economics']
    tech = params['technology']
    costs = params['costs']   # 👉 关键：这里加上
    
    discount_factor = [(1 + econ['r']) ** (-t) for t in model.T]

    # 收入
    revenue = quicksum(
        econ['P_m'] * model.Qt[t] * discount_factor[t]
        for t in model.T
    )
    
    # 成本项
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
    """计算详细的成本分解"""
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
    """打印成本分解"""
    print(f"\n经济分析")
    print(f"{'─'*40}")
    
    # 收入部分
    print(f"收入")
    print(f"  总收入            : ${costs['revenue']:>15,.2f}")
    
    # 成本部分 - 按逻辑分组
    print(f"\n成本构成")
    print(f"  开发成本          : ${costs['development_cost']:>15,.2f}")
    print(f"  运营成本          : ${costs['operating_cost']:>15,.2f}")
    print(f"  发射成本          : ${costs['launch_cost']:>15,.2f}")
    print(f"  存储成本          : ${costs['storage_cost']:>15,.2f}")
    print(f"  副产物成本        : ${costs['byproduct_cost']:>15,.2f}")
    print(f"  地球供氧成本      : ${costs['earth_supply_cost']:>15,.2f}")
    
    # 惩罚成本（如果有）
    shortage_cost = costs.get('shortage_cost', 0)
    if shortage_cost > 0:
        print(f"  短缺惩罚          : ${shortage_cost:>15,.2f}")
    
    build_cost = costs.get('build_cost', 0)
    if build_cost > 0:
        print(f"  建设成本          : ${build_cost:>15,.2f}")
    
    # 汇总
    print(f"  {'─'*32}")
    print(f"  总成本            : ${costs['total_cost']:>15,.2f}")
    
    # 净现值
    print(f"\n最终结果")
    print(f"  净现值 (NPV)      : ${costs['NPV']:>15,.2f}")

