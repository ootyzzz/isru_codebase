#!/usr/bin/env python3
"""
系统状态管理器
管理ISRU系统的状态信息和状态转换
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class SystemState:
    """系统状态数据结构"""
    # 时间信息
    current_time: int = 0
    
    # 产能信息
    total_capacity: float = 0.0          # 总产能 [kg/year]
    deployed_mass: float = 0.0           # 已部署ISRU质量 [kg]
    
    # 需求和生产
    current_demand: float = 0.0          # 当前需求 [kg]
    actual_production: float = 0.0       # 实际产量 [kg]
    
    # 库存管理
    inventory: float = 0.0               # 当前库存 [kg]
    
    # 补给信息
    earth_supply: float = 0.0            # 地球补给量 [kg]
    
    # 历史记录
    demand_history: List[float] = field(default_factory=list)
    production_history: List[float] = field(default_factory=list)
    capacity_history: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """计算衍生指标"""
        self.update_derived_metrics()
    
    def update_derived_metrics(self):
        """更新衍生指标"""
        # 利用率
        if self.total_capacity > 0:
            self.utilization_rate = min(self.current_demand / self.total_capacity, 1.0)
        else:
            self.utilization_rate = 0.0
        
        # 库存水平（相对于需求的天数）
        if self.current_demand > 0:
            self.inventory_days = self.inventory / (self.current_demand / 365)
        else:
            self.inventory_days = float('inf') if self.inventory > 0 else 0.0
        
        # 供需平衡
        self.supply_demand_ratio = (self.actual_production + self.earth_supply) / max(self.current_demand, 1e-6)
        
        # 自给自足率
        self.self_sufficiency_rate = self.actual_production / max(self.current_demand, 1e-6)


@dataclass
class Decision:
    """决策结果数据结构"""
    time: int
    
    # 产能决策
    capacity_expansion: float = 0.0      # 新增产能 [kg/year]
    new_deployment: float = 0.0          # 新增部署质量 [kg]
    
    # 生产决策
    planned_production: float = 0.0      # 计划产量 [kg]
    
    # 补给决策
    earth_supply_request: float = 0.0    # 地球补给请求 [kg]
    
    # 库存决策
    inventory_target: float = 0.0        # 目标库存 [kg]
    
    # 决策原因
    decision_reason: str = ""
    
    # 成本估算
    expansion_cost: float = 0.0          # 扩张成本
    operational_cost: float = 0.0        # 运营成本
    supply_cost: float = 0.0             # 补给成本


class StateManager:
    """系统状态管理器"""
    
    def __init__(self, initial_state: Optional[SystemState] = None):
        """初始化状态管理器"""
        self.state = initial_state or SystemState()
        self.decision_history: List[Decision] = []
        
    def update_state(self, decision: Decision, new_demand: float, params: Dict) -> SystemState:
        """
        根据决策和新需求更新系统状态
        
        Args:
            decision: 当前决策
            new_demand: 新的需求
            params: 系统参数
            
        Returns:
            更新后的状态
        """
        # 更新时间
        self.state.current_time += 1
        
        # 更新需求
        self.state.current_demand = new_demand
        self.state.demand_history.append(new_demand)
        
        # 执行产能扩张
        if decision.capacity_expansion > 0:
            self.state.total_capacity += decision.capacity_expansion
            self.state.deployed_mass += decision.new_deployment
        
        # 计算实际产量（受产能限制）
        max_production = self.state.total_capacity
        self.state.actual_production = min(decision.planned_production, max_production)
        self.state.production_history.append(self.state.actual_production)
        self.state.capacity_history.append(self.state.total_capacity)
        
        # 计算地球补给需求
        production_shortfall = max(0, self.state.current_demand - self.state.actual_production)
        self.state.earth_supply = min(decision.earth_supply_request, production_shortfall)
        
        # 更新库存
        inventory_change = (self.state.actual_production + self.state.earth_supply - 
                          self.state.current_demand)
        self.state.inventory = max(0, self.state.inventory + inventory_change)
        
        # 更新衍生指标
        self.state.update_derived_metrics()
        
        # 记录决策
        self.decision_history.append(decision)
        
        return self.state
    
    def get_demand_trend(self, window: int = 3) -> Optional[List[float]]:
        """获取需求趋势"""
        if len(self.state.demand_history) < window:
            return None
        return self.state.demand_history[-window:]
    
    def get_utilization_trend(self, window: int = 3) -> List[float]:
        """获取利用率趋势"""
        if len(self.state.demand_history) < window or len(self.state.capacity_history) < window:
            return []
        
        utilizations = []
        for i in range(-window, 0):
            demand = self.state.demand_history[i]
            capacity = self.state.capacity_history[i]
            util = min(demand / max(capacity, 1e-6), 1.0)
            utilizations.append(util)
        
        return utilizations
    
    def calculate_performance_metrics(self, params: Dict) -> Dict[str, float]:
        """
        计算性能指标
        
        Args:
            params: 系统参数
            
        Returns:
            性能指标字典
        """
        if not self.decision_history:
            return {}
        
        # 基本统计
        total_demand = sum(self.state.demand_history)
        total_production = sum(self.state.production_history)
        total_earth_supply = sum(d.earth_supply_request for d in self.decision_history)
        
        # 成本计算
        total_expansion_cost = sum(d.expansion_cost for d in self.decision_history)
        total_operational_cost = sum(d.operational_cost for d in self.decision_history)
        total_supply_cost = sum(d.supply_cost for d in self.decision_history)
        total_cost = total_expansion_cost + total_operational_cost + total_supply_cost
        
        # 收入计算（假设每kg氧气价格）
        oxygen_price = params.get('costs', {}).get('P_m', 21160)
        total_revenue = total_production * oxygen_price
        
        # NPV计算（简化版）
        discount_rate = params.get('economics', {}).get('r', 0.1)
        npv = 0
        for i, decision in enumerate(self.decision_history):
            year_revenue = self.state.production_history[i] * oxygen_price
            year_cost = decision.expansion_cost + decision.operational_cost + decision.supply_cost
            year_cash_flow = year_revenue - year_cost
            npv += year_cash_flow / ((1 + discount_rate) ** (i + 1))
        
        # 利用率统计
        utilizations = []
        for i in range(len(self.state.demand_history)):
            if i < len(self.state.capacity_history):
                util = min(self.state.demand_history[i] / max(self.state.capacity_history[i], 1e-6), 1.0)
                utilizations.append(util)
        
        return {
            # 财务指标
            'npv': npv,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'total_expansion_cost': total_expansion_cost,
            'total_operational_cost': total_operational_cost,
            'total_supply_cost': total_supply_cost,
            
            # 运营指标
            'total_demand': total_demand,
            'total_production': total_production,
            'total_earth_supply': total_earth_supply,
            'self_sufficiency_rate': total_production / max(total_demand, 1e-6),
            'avg_utilization': np.mean(utilizations) if utilizations else 0,
            'min_utilization': min(utilizations) if utilizations else 0,
            'max_utilization': max(utilizations) if utilizations else 0,
            'utilization_std': np.std(utilizations) if utilizations else 0,
            
            # 容量指标
            'final_capacity': self.state.total_capacity,
            'final_deployed_mass': self.state.deployed_mass,
            'capacity_expansions': sum(1 for d in self.decision_history if d.capacity_expansion > 0),
            
            # 库存指标
            'final_inventory': self.state.inventory,
            'avg_inventory_days': np.mean([self.state.inventory_days]) if hasattr(self.state, 'inventory_days') else 0
        }
    
    def get_state_summary(self) -> Dict[str, float]:
        """获取当前状态摘要"""
        return {
            'time': self.state.current_time,
            'total_capacity': self.state.total_capacity,
            'current_demand': self.state.current_demand,
            'utilization_rate': self.state.utilization_rate,
            'inventory': self.state.inventory,
            'self_sufficiency_rate': self.state.self_sufficiency_rate,
            'deployed_mass': self.state.deployed_mass
        }


if __name__ == "__main__":
    # 测试代码
    print("=== 状态管理器测试 ===")
    
    # 创建初始状态
    initial_state = SystemState(
        total_capacity=100.0,
        deployed_mass=500.0,
        current_demand=80.0
    )
    
    manager = StateManager(initial_state)
    print(f"初始状态: {manager.get_state_summary()}")
    
    # 模拟一个决策
    decision = Decision(
        time=1,
        capacity_expansion=20.0,
        new_deployment=100.0,
        planned_production=90.0,
        earth_supply_request=10.0,
        decision_reason="需求增长，扩张产能"
    )
    
    # 更新状态
    params = {'economics': {'r': 0.1}, 'costs': {'P_m': 21160}}
    new_state = manager.update_state(decision, new_demand=110.0, params=params)
    
    print(f"更新后状态: {manager.get_state_summary()}")
    
    # 计算性能指标
    metrics = manager.calculate_performance_metrics(params)
    print(f"性能指标: {metrics}")