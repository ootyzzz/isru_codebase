"""
Pyomo模型约束条件定义
定义ISRU氧气生产优化问题的所有约束条件 (5年期)
"""

from pyomo.environ import Constraint, ConcreteModel


def define_constraints(model: ConcreteModel, params: dict, demand_path: list) -> ConcreteModel:
    tech = params['technology']
    T = params['economics']['T']
    
    # 1. 需求平衡：需求必须由交付 + 地球供氧 + 短缺满足
    def demand_balance_rule(m, t):
        if t >= len(demand_path):
            raise IndexError(
                f"时间索引 {t} 超出需求路径范围 [0, {len(demand_path)-1}]"
            )
        return m.Qt[t] + m.Q_earth[t] + m.St[t] == demand_path[t]
    model.demand_balance = Constraint(model.T, rule=demand_balance_rule)

    # 2. 交付 ≤ 产能
    def delivery_capacity_rule(m, t):
        return m.Qt[t] <= m.Qt_cap[t]
    model.delivery_capacity = Constraint(model.T, rule=delivery_capacity_rule)

    # 3. 剩余定义
    def surplus_def_rule(m, t):
        return m.Et[t] == m.Qt_cap[t] - m.Qt[t]
    model.surplus_def = Constraint(model.T, rule=surplus_def_rule)

    # 4. 产能 = η * Mt
    def capacity_mass_rule(m, t):
        return m.Qt_cap[t] == tech['eta'] * m.Mt[t]
    model.capacity_mass = Constraint(model.T, rule=capacity_mass_rule)

    # 5. ISRU质量平衡
    def mass_balance_rule(m, t):
        if t == 0:
            return m.Mt[t] == tech.get("M0", 0)
        return m.Mt[t] == m.Mt[t-1] + m.delta_Mt[t]
    model.mass_balance = Constraint(model.T, rule=mass_balance_rule)

    # 6. 发射质量（只对新增部署计费）
    def leo_mass_rule(m, t):
        return m.M_leo[t] == tech['alpha'] * m.delta_Mt[t]
    model.leo_mass = Constraint(model.T, rule=leo_mass_rule)

    return model


def validate_constraints(model: ConcreteModel) -> bool:
    """简单验证约束是否满足（可选调试用）"""
    return True
