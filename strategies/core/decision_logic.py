#!/usr/bin/env python3
"""
决策逻辑引擎
实现三种策略的具体决策逻辑
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.strategy_definitions import StrategyParams, StrategyType
from strategies.core.state_manager import SystemState, Decision


class DecisionEngine:
    """策略决策引擎"""
    
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
    
    def make_decision(self, state: SystemState, demand_forecast: Optional[List[float]] = None) -> Decision:
        """
        根据策略和当前状态做出决策
        
        Args:
            state: 当前系统状态
            demand_forecast: 需求预测（可选）
            
        Returns:
            决策结果
        """
        decision = Decision(time=state.current_time)
        
        # 1. 评估是否需要扩张产能
        expansion_needed, expansion_reason = self._evaluate_expansion_need(state, demand_forecast)
        
        if expansion_needed:
            # 2. 计算扩张规模
            expansion_amount = self._calculate_expansion_amount(state, demand_forecast)
            decision.capacity_expansion = expansion_amount
            decision.new_deployment = expansion_amount / self.efficiency_ratio
            decision.expansion_cost = self._calculate_expansion_cost(expansion_amount)
            decision.decision_reason = expansion_reason
        
        # 3. 计算生产计划
        decision.planned_production = self._calculate_production_plan(state)
        decision.operational_cost = self._calculate_operational_cost(decision.planned_production)
        
        # 4. 计算地球补给需求
        production_shortfall = max(0, state.current_demand - decision.planned_production)
        decision.earth_supply_request = self._calculate_earth_supply(production_shortfall, state)
        decision.supply_cost = self._calculate_supply_cost(decision.earth_supply_request)
        
        # 5. 设置库存目标
        decision.inventory_target = self._calculate_inventory_target(state)
        
        return decision
    
    def _evaluate_expansion_need(self, state: SystemState, 
                               demand_forecast: Optional[List[float]] = None) -> tuple[bool, str]:
        """评估是否需要扩张产能"""
        
        # 基本利用率检查
        if state.utilization_rate < self.strategy.utilization_threshold:
            return False, "利用率未达到扩张阈值"
        
        # 策略特定的扩张逻辑
        if self.strategy.name == "conservative":
            return self._conservative_expansion_logic(state, demand_forecast)
        elif self.strategy.name == "aggressive":
            return self._aggressive_expansion_logic(state, demand_forecast)
        elif self.strategy.name == "moderate":
            return self._moderate_expansion_logic(state, demand_forecast)
        else:
            return False, "未知策略类型"
    
    def _conservative_expansion_logic(self, state: SystemState, 
                                    demand_forecast: Optional[List[float]] = None) -> tuple[bool, str]:
        """保守策略的扩张逻辑"""
        
        # 需要连续高利用率
        if len(state.demand_history) >= 2:
            recent_utilizations = []
            for i in range(-2, 0):
                if abs(i) <= len(state.demand_history) and abs(i) <= len(state.capacity_history):
                    demand = state.demand_history[i]
                    capacity = state.capacity_history[i] if abs(i) <= len(state.capacity_history) else state.total_capacity
                    util = min(demand / max(capacity, 1e-6), 1.0)
                    recent_utilizations.append(util)
            
            if recent_utilizations and min(recent_utilizations) < self.strategy.utilization_threshold:
                return False, "保守策略：需要连续高利用率才扩张"
        
        # 检查需求增长趋势
        if len(state.demand_history) >= 3:
            recent_demands = state.demand_history[-3:]
            growth_rates = [recent_demands[i+1]/recent_demands[i] - 1 for i in range(len(recent_demands)-1)]
            avg_growth = np.mean(growth_rates)
            
            if avg_growth <= 0.05:  # 增长率低于5%
                return False, "保守策略：需求增长趋势不明显"
        
        return True, "保守策略：满足连续高利用率和需求增长条件"
    
    def _aggressive_expansion_logic(self, state: SystemState, 
                                  demand_forecast: Optional[List[float]] = None) -> tuple[bool, str]:
        """激进策略的扩张逻辑"""
        
        # 激进策略：只要利用率达到阈值就扩张
        if state.utilization_rate >= self.strategy.utilization_threshold:
            return True, "激进策略：利用率达到阈值，立即扩张"
        
        # 预测性扩张：如果预测未来需求会增长
        if demand_forecast and len(demand_forecast) > 0:
            future_demand = demand_forecast[0]
            if future_demand > state.current_demand * 1.1:  # 预测增长超过10%
                return True, "激进策略：预测需求增长，提前扩张"
        
        return False, "激进策略：条件不满足"
    
    def _moderate_expansion_logic(self, state: SystemState, 
                                demand_forecast: Optional[List[float]] = None) -> tuple[bool, str]:
        """温和策略的扩张逻辑"""
        
        # 温和策略：平衡考虑当前利用率和未来趋势
        if state.utilization_rate >= self.strategy.utilization_threshold:
            
            # 检查库存水平
            if hasattr(state, 'inventory_days') and state.inventory_days < 30:  # 库存不足30天
                return True, "温和策略：高利用率且库存不足"
            
            # 检查最近的需求趋势
            if len(state.demand_history) >= 2:
                recent_growth = (state.demand_history[-1] / state.demand_history[-2] - 1) if state.demand_history[-2] > 0 else 0
                if recent_growth > 0.02:  # 最近增长超过2%
                    return True, "温和策略：高利用率且需求有增长趋势"
        
        return False, "温和策略：条件不满足"
    
    def _calculate_expansion_amount(self, state: SystemState, 
                                  demand_forecast: Optional[List[float]] = None) -> float:
        """计算扩张数量"""
        
        # 基础扩张量
        base_expansion = state.total_capacity * self.strategy.expansion_ratio
        
        # 当前需求缺口
        current_gap = max(0, state.current_demand - state.total_capacity)
        
        # 策略特定的扩张计算
        if self.strategy.name == "conservative":
            # 保守策略：只满足当前缺口的80%，避免过度投资
            expansion = min(base_expansion, current_gap * 0.8)
            
        elif self.strategy.name == "aggressive":
            # 激进策略：不仅满足当前缺口，还要预留未来增长空间
            future_buffer = state.current_demand * 0.3  # 30%的未来缓冲
            expansion = max(base_expansion, current_gap + future_buffer)
            
        else:  # moderate
            # 温和策略：满足当前缺口并适度预留
            future_buffer = state.current_demand * 0.15  # 15%的未来缓冲
            expansion = max(base_expansion, current_gap + future_buffer)
        
        # 确保扩张量不为负
        return max(0, expansion)
    
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
    
    def _calculate_earth_supply(self, production_shortfall: float, state: SystemState) -> float:
        """计算地球补给需求"""
        if production_shortfall <= 0:
            return 0
        
        # 策略特定的补给逻辑
        if self.strategy.name == "conservative":
            # 保守策略：优先使用地球补给，避免过度投资
            return production_shortfall
            
        elif self.strategy.name == "aggressive":
            # 激进策略：尽量减少地球补给，推动本地生产
            # 只在紧急情况下使用地球补给
            emergency_threshold = state.current_demand * 0.2  # 20%的紧急阈值
            return min(production_shortfall, emergency_threshold)
            
        else:  # moderate
            # 温和策略：平衡使用地球补给
            return production_shortfall * 0.8  # 补给80%的缺口
    
    def _calculate_inventory_target(self, state: SystemState) -> float:
        """计算库存目标"""
        # 基于策略的风险偏好设置库存目标
        base_target = state.current_demand * 0.1  # 基础目标：10%的年需求
        
        # 根据风险容忍度调整
        risk_multiplier = 1 + (1 - self.strategy.risk_tolerance)
        
        return base_target * risk_multiplier
    
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
    print("=== 决策逻辑引擎测试 ===")
    
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
        }
    }
    
    # 测试每种策略
    strategies = StrategyDefinitions.get_all_strategies()
    
    for name, strategy in strategies.items():
        print(f"\n=== {name.upper()} 策略测试 ===")
        
        engine = DecisionEngine(strategy, params)
        
        # 创建测试状态
        test_state = SystemState(
            current_time=5,
            total_capacity=100.0,
            current_demand=90.0,  # 90%利用率
            inventory=10.0,
            demand_history=[70, 75, 80, 85, 90],
            capacity_history=[100, 100, 100, 100, 100]
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