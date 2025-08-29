#!/usr/bin/env python3
"""
决策逻辑引擎 - 生产策略版本
实现三种新生产策略的具体决策逻辑
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.strategy_definitions import StrategyParams, StrategyType, calculate_production_for_year, calculate_required_capacity
from strategies.core.state_manager import SystemState, Decision


class DecisionEngine:
    """生产策略决策引擎"""
    
    def __init__(self, strategy: StrategyParams, params: Dict):
        """
        初始化决策引擎
        
        Args:
            strategy: 策略参数
            params: 系统参数
        """
        self.strategy = strategy
        self.params = params
        
        # 从参数中提取成本信息
        costs = params.get('costs', {})
        self.capacity_cost_per_kg = costs.get('c_dev', 10000)  # 产能建设成本
        self.operational_cost_per_kg = costs.get('c_op', 3000)  # 运营成本
        self.earth_supply_cost_per_kg = costs.get('c_E', 20000)  # 地球补给成本
        
        # 技术参数
        tech = params.get('technology', {})
        self.efficiency_ratio = tech.get('eta', 2)  # 产能效率比
        
        # 需求参数
        demand_params = params.get('demand', {})
        self.D0 = demand_params.get('D0', 10)  # 初始需求
        self.mu = demand_params.get('mu', 0.2)  # 需求增长率
        self.sigma = demand_params.get('sigma', 0.2)  # 需求波动率
    
    def make_decision(self, state: SystemState, demand_forecast: Optional[List[float]] = None) -> Decision:
        """
        根据生产策略和当前状态做出决策
        
        Args:
            state: 当前系统状态
            demand_forecast: 需求预测（可选）
            
        Returns:
            决策结果
        """
        decision = Decision(time=state.current_time)
        
        # 1. 计算最终需求（基于时间跨度T）
        final_demand = self._estimate_final_demand(state)
        
        # 2. 根据策略计算当年的目标生产量
        target_production = self._calculate_target_production(state, final_demand)
        
        # 3. 计算所需产能
        required_capacity = calculate_required_capacity(self.strategy, target_production)
        
        # 4. 检查是否需要扩张产能
        capacity_gap = max(0, required_capacity - state.total_capacity)
        
        if capacity_gap > 0:
            decision.capacity_expansion = capacity_gap
            decision.new_deployment = capacity_gap / self.efficiency_ratio
            decision.expansion_cost = self._calculate_expansion_cost(capacity_gap)
            decision.decision_reason = f"{self.strategy.production_type} production strategy capacity expansion"
        
        # 5. 设置生产计划
        decision.planned_production = min(target_production, state.total_capacity + capacity_gap)
        decision.operational_cost = self._calculate_operational_cost(decision.planned_production)
        
        # 6. 计算地球补给需求
        production_shortfall = max(0, state.current_demand - decision.planned_production)
        decision.earth_supply_request = production_shortfall
        decision.supply_cost = self._calculate_supply_cost(decision.earth_supply_request)
        
        # 7. 设置库存目标
        decision.inventory_target = self._calculate_inventory_target(state)
        
        return decision
    
    def _estimate_final_demand(self, state: SystemState) -> float:
        """
        估算最终年份的需求
        
        Args:
            state: 当前系统状态
            
        Returns:
            最终年份的预期需求
        """
        # 使用几何布朗运动模型估算最终需求
        # D_T = D0 * exp(mu*T)
        T = self.strategy.time_horizon
        final_demand = self.D0 * np.exp(self.mu * T)
        
        return final_demand
    
    def _calculate_target_production(self, state: SystemState, final_demand: float) -> float:
        """
        根据策略计算当年的目标生产量
        
        Args:
            state: 当前系统状态
            final_demand: 最终需求
            
        Returns:
            当年的目标生产量
        """
        current_year = state.current_time
        current_demand = state.current_demand
        
        return calculate_production_for_year(
            self.strategy,
            current_year,
            final_demand,
            current_demand
        )
    
    def _calculate_inventory_target(self, state: SystemState) -> float:
        """计算库存目标"""
        # 基础目标：10%的年需求
        base_target = state.current_demand * 0.1
        
        # 根据策略类型调整
        if self.strategy.production_type == "upfront":
            # 一次性满产策略：保持较高库存
            return base_target * 1.5
        elif self.strategy.production_type == "gradual":
            # 渐进生产策略：中等库存
            return base_target * 1.2
        else:  # flexible
            # 灵活生产策略：较低库存，依赖快速响应
            return base_target * 0.8
    
    def _calculate_expansion_cost(self, expansion_amount: float) -> float:
        """计算扩张成本"""
        # 产能建设成本
        capacity_cost = expansion_amount * self.capacity_cost_per_kg
        
        # 规模经济效应
        if expansion_amount > 100:  # 大规模扩张有折扣
            capacity_cost *= 0.9
        
        return capacity_cost
    
    def _calculate_operational_cost(self, production_amount: float) -> float:
        """计算运营成本"""
        return production_amount * self.operational_cost_per_kg
    
    def _calculate_supply_cost(self, supply_amount: float) -> float:
        """计算地球补给成本"""
        return supply_amount * self.earth_supply_cost_per_kg


if __name__ == "__main__":
    # 测试代码
    print("=== 生产策略决策逻辑引擎测试 ===")
    
    # 导入策略定义
    from strategies.core.strategy_definitions import StrategyDefinitions
    
    # 测试参数
    params = {
        'costs': {
            'c_dev': 10000,
            'c_op': 3000,
            'c_E': 20000
        },
        'technology': {
            'eta': 2
        },
        'demand': {
            'D0': 10,
            'mu': 0.2,
            'sigma': 0.2
        }
    }
    
    # 测试每种新策略
    time_horizon = 20
    strategies = StrategyDefinitions.get_all_strategies(time_horizon)
    
    for name, strategy in strategies.items():
        print(f"\n=== {name.upper()} 策略测试 ===")
        
        engine = DecisionEngine(strategy, params)
        
        # 创建测试状态
        test_state = SystemState(
            current_time=5,
            total_capacity=50.0,
            current_demand=60.0,  # 需求超过产能
            inventory=5.0,
            demand_history=[40, 45, 50, 55, 60],
            capacity_history=[30, 35, 40, 45, 50]
        )
        
        # 做出决策
        decision = engine.make_decision(test_state)
        
        print(f"决策结果:")
        print(f"  目标生产量: {decision.planned_production:.1f}")
        print(f"  产能扩张: {decision.capacity_expansion:.1f}")
        print(f"  新增部署: {decision.new_deployment:.1f}")
        print(f"  地球补给: {decision.earth_supply_request:.1f}")
        print(f"  扩张成本: ${decision.expansion_cost:,.0f}")
        print(f"  决策原因: {decision.decision_reason}")