"""
Pyomo模型决策变量定义
定义ISRU氧气生产优化问题的所有决策变量
"""

from pyomo.environ import ConcreteModel, Var, NonNegativeReals, RangeSet


def define_variables(model: ConcreteModel, params: dict) -> ConcreteModel:
    """
    定义优化模型的决策变量
    
    Args:
        model: Pyomo模型实例
        params: 参数字典
        
    Returns:
        包含变量的Pyomo模型
    """
    T = params['economics']['T']
    
    # 时间索引
    model.T = RangeSet(0, T)
    
    # 决策变量
    # Qt: 交付的氧气量 [kg]
    model.Qt = Var(model.T, within=NonNegativeReals, doc="Delivered oxygen quantity")
    
    # Qt_cap: 生产能力 [kg]
    model.Qt_cap = Var(model.T, within=NonNegativeReals, doc="Production capacity")
    
    # St: 短缺量 [kg]
    model.St = Var(model.T, within=NonNegativeReals, doc="Shortage quantity")
    
    # Et: 剩余量 [kg]
    model.Et = Var(model.T, within=NonNegativeReals, doc="Excess quantity")
    
    # Mt: 部署的ISRU质量 [kg]
    model.Mt = Var(model.T, within=NonNegativeReals, doc="Deployed ISRU mass")
    
    # delta_Mt: 新增ISRU部署 [kg]
    model.delta_Mt = Var(model.T, within=NonNegativeReals, doc="New ISRU deployment")
    
    # M_leo: 发射到LEO的质量 [kg]
    model.M_leo = Var(model.T, within=NonNegativeReals, doc="Mass launched at LEO")

    # Q_earth: 从地球发射的氧气量 [kg]
    model.Q_earth = Var(model.T, within=NonNegativeReals, doc="Earth-supplied oxygen quantity")

    return model


def get_variable_names() -> dict:
    """返回所有决策变量的名称映射"""
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
    """返回变量的合理边界"""
    max_demand = params['demand']['D0'] * 5  # 最大需求估计
    
    return {
        'Qt': (0, max_demand),
        'Qt_cap': (0, max_demand * 2),
        'St': (0, max_demand),
        'Et': (0, max_demand),
        'Mt': (0, max_demand * 1000),        # 考虑质量转换
        'delta_Mt': (0, max_demand * 1000),
        'M_leo': (0, max_demand * 1000),      # 与 Mt 同阶
        'Q_earth': (0, max_demand * 2)
    }
