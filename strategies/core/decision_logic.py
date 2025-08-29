#!/usr/bin/env python3
"""
决策逻辑引擎 - 新策略版本
实现三种新部署策略的具体决策逻辑
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.strategy_definitions import StrategyParams, StrategyType, calculate_deployment_for_year
from strategies.core.state_manager import SystemState, Decision


class DecisionEngine:
    """新策略决策引擎"""
    
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
        根据新策略和当前状态做出决策
        
        Args:
            state: 当前系统状态
            demand_forecast: 需求预测（可选）
            
        Returns:
            决策结果
        """
        decision = Decision(time=state.current_time)
        
        # 1. 计算最终需求（基于时间跨度T）
        final_demand = self._estimate_final_demand(state)
        
        # 2. 根据策略计算当年的部署量
        deployment_amount = self._calculate_deployment_amount(state, final_demand)
        
        if deployment_amount > 0:
            decision.capacity_expansion = deployment_amount
            decision.new_deployment = deployment_amount / self.efficiency_ratio
            decision.expansion_cost = self._calculate_expansion_cost(deployment_amount)
            decision.decision_reason = f"{self.strategy.deployment_type} strategy deployment"
        
        # 3. 计算生产计划
        decision.planned_production = self._calculate_production_plan(state)
        decision.operational_cost = self._calculate_operational_cost(decision.planned_production)
        
        # 4. 计算地球补给需求
        production_shortfall = max(0, state.current_demand - decision.planned_production)
        decision.earth_supply_request = production_shortfall
        decision.supply_cost = self._calculate_supply_cost(decision.earth_supply_request)
        
        # 5. 设置库存目标
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
        # D_T = D0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        # 这里使用期望值：D_T = D0 * exp(mu*T)
        
        T = self.strategy.time_horizon
        final_demand = self.D0 * np.exp(self.mu * T)
        
        return final_demand
    
    def _calculate_deployment_amount(self, state: SystemState, final_demand: float) -> float:
        """
        根据策略计算当年的部署量
        
        Args:
            state: 当前系统状态
            final_demand: 最终需求
            
        Returns:
            当年的新增部署量
        """
        current_year = state.current_time
        
        # 获取上一年的供需信息（用于灵活策略）
        previous_supply = 0
        previous_demand = 0
        
        if len(state.demand_history) > 0 and len(state.capacity_history) > 0:
            previous_demand = state.demand_history[-1]
            previous_capacity = state.capacity_history[-1] if state.capacity_history else 0
            # 假设上一年的供应量等于产能（简化）
            previous_supply = previous_capacity
        
        return calculate_deployment_for_year(
            self.strategy,
            current_year,
            final_demand,
            previous_supply,
            previous_demand
        )
    
    def _calculate_production_plan(self, state: SystemState) -> float:
        """计算生产计划"""
        # 生产计划受产能限制
        max_production = state.total_capacity
        
        # 根据需求和库存情况调整生产计划
        target_production = state.current_demand
        
        # 考虑库存情况
        if state.inventory > state.current_demand * 0.1:  # 库存超过10%的年需求
            # 减少生产，避免库存积压
            target_production *= 0.9
        
        return min(target_production, max_production)
    
    def _calculate_inventory_target(self, state: SystemState) -> float:
        """计算库存目标"""
        # 基础目标：10%的年需求
        base_target = state.current_demand * 0.1
        
        # 根据策略类型调整
        if self.strategy.deployment_type == "upfront":
            # 一次性部署策略：保持较高库存
            return base_target * 1.5
        elif self.strategy.deployment_type == "gradual":
            # 渐进部署策略：中等库存
            return base_target * 1.2
        else:  # flexible
            # 灵活部署策略：较低库存，依赖快速响应
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
    print("=== 新策略决策逻辑引擎测试 ===")
    
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
        print(f"  产能扩张: {decision.capacity_expansion:.1f}")
        print(f"  新增部署: {decision.new_deployment:.1f}")
        print(f"  计划产量: {decision.planned_production:.1f}")
        print(f"  地球补给: {decision.earth_supply_request:.1f}")
        print(f"  扩张成本: ${decision.expansion_cost:,.0f}")
        print(f"  决策原因: {decision.decision_reason}")