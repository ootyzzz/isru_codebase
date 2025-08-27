#!/usr/bin/env python3
"""
ISRU策略定义模块 - 重构版本
定义三种策略的参数和决策逻辑
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class StrategyType(Enum):
    """策略类型枚举"""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"


@dataclass
class StrategyParams:
    """策略参数数据结构"""
    name: str
    description: str
    
    # 初始部署参数
    initial_deployment_ratio: float  # 相对于预期总需求的初始部署比例
    
    # 决策触发参数
    utilization_threshold: float     # 触发扩张的利用率阈值
    expansion_ratio: float           # 每次扩张的比例
    
    # 风险偏好参数
    risk_tolerance: float            # 风险容忍度 (0-1)
    cost_sensitivity: float          # 成本敏感度 (0-1)
    
    def __post_init__(self):
        """验证参数有效性"""
        assert 0 <= self.initial_deployment_ratio <= 2.0, "初始部署比例应在0-2.0之间"
        assert 0.5 <= self.utilization_threshold <= 1.0, "利用率阈值应在0.5-1.0之间"
        assert 0 <= self.expansion_ratio <= 1.0, "扩张比例应在0-1.0之间"
        assert 0 <= self.risk_tolerance <= 1.0, "风险容忍度应在0-1.0之间"
        assert 0 <= self.cost_sensitivity <= 1.0, "成本敏感度应在0-1.0之间"


class StrategyDefinitions:
    """策略定义类 - 重构版本"""
    
    @staticmethod
    def get_conservative_strategy() -> StrategyParams:
        """
        保守策略：谨慎扩张，重视成本控制
        - 初期部署较少，避免过度投资
        - 高利用率才扩张，确保需求确实存在
        - 小幅扩张，控制风险
        """
        return StrategyParams(
            name="conservative",
            description="保守策略：谨慎扩张，重视成本控制",
            initial_deployment_ratio=0.6,   # 初期只部署60%的预期需求
            utilization_threshold=0.85,     # 85%利用率才扩张
            expansion_ratio=0.25,           # 每次扩张25%
            risk_tolerance=0.3,             # 低风险容忍度
            cost_sensitivity=0.8            # 高成本敏感度
        )
    
    @staticmethod
    def get_aggressive_strategy() -> StrategyParams:
        """
        激进策略：快速扩张，追求市场占有
        - 初期大规模部署，抢占市场
        - 较低利用率就扩张，提前布局
        - 大幅扩张，快速响应需求增长
        """
        return StrategyParams(
            name="aggressive",
            description="激进策略：快速扩张，追求市场占有",
            initial_deployment_ratio=1.2,   # 初期部署120%的预期需求
            utilization_threshold=0.70,     # 70%利用率就扩张
            expansion_ratio=0.50,           # 每次扩张50%
            risk_tolerance=0.8,             # 高风险容忍度
            cost_sensitivity=0.3            # 低成本敏感度
        )
    
    @staticmethod
    def get_moderate_strategy() -> StrategyParams:
        """
        温和策略：平衡扩张，稳健发展
        - 中等规模初期部署
        - 中等利用率触发扩张
        - 中等幅度扩张
        """
        return StrategyParams(
            name="moderate",
            description="温和策略：平衡扩张，稳健发展",
            initial_deployment_ratio=0.9,   # 初期部署90%的预期需求
            utilization_threshold=0.80,     # 80%利用率扩张
            expansion_ratio=0.35,           # 每次扩张35%
            risk_tolerance=0.5,             # 中等风险容忍度
            cost_sensitivity=0.5            # 中等成本敏感度
        )
    
    @classmethod
    def get_strategy(cls, strategy_type: StrategyType) -> StrategyParams:
        """根据策略类型获取策略参数"""
        strategy_map = {
            StrategyType.CONSERVATIVE: cls.get_conservative_strategy,
            StrategyType.AGGRESSIVE: cls.get_aggressive_strategy,
            StrategyType.MODERATE: cls.get_moderate_strategy
        }
        
        if strategy_type not in strategy_map:
            raise ValueError(f"未知的策略类型: {strategy_type}")
        
        return strategy_map[strategy_type]()
    
    @classmethod
    def get_all_strategies(cls) -> Dict[str, StrategyParams]:
        """获取所有策略"""
        return {
            "conservative": cls.get_conservative_strategy(),
            "aggressive": cls.get_aggressive_strategy(),
            "moderate": cls.get_moderate_strategy()
        }


def calculate_initial_capacity(strategy: StrategyParams, 
                             expected_total_demand: float) -> float:
    """
    根据策略计算初始产能部署
    
    Args:
        strategy: 策略参数
        expected_total_demand: 预期总需求
        
    Returns:
        初始产能
    """
    return strategy.initial_deployment_ratio * expected_total_demand


def should_expand_capacity(strategy: StrategyParams, 
                          current_utilization: float,
                          current_capacity: float,
                          demand_trend: Optional[List[float]] = None) -> bool:
    """
    判断是否应该扩张产能
    
    Args:
        strategy: 策略参数
        current_utilization: 当前利用率
        current_capacity: 当前产能
        demand_trend: 需求趋势（可选）
        
    Returns:
        是否应该扩张
    """
    # 基本判断：利用率是否超过阈值
    if current_utilization < strategy.utilization_threshold:
        return False
    
    # 对于保守策略，需要更严格的条件
    if strategy.name == "conservative":
        # 需要连续高利用率才扩张
        if demand_trend and len(demand_trend) >= 3:
            recent_growth = np.mean(np.diff(demand_trend[-3:]))
            if recent_growth <= 0:  # 需求没有增长趋势
                return False
    
    return True


def calculate_expansion_amount(strategy: StrategyParams,
                             current_capacity: float,
                             current_demand: float) -> float:
    """
    计算扩张数量
    
    Args:
        strategy: 策略参数
        current_capacity: 当前产能
        current_demand: 当前需求
        
    Returns:
        扩张数量
    """
    base_expansion = current_capacity * strategy.expansion_ratio
    
    # 根据需求缺口调整扩张量
    demand_gap = max(0, current_demand - current_capacity)
    
    if strategy.name == "aggressive":
        # 激进策略：不仅满足当前缺口，还要预留更多产能
        expansion = max(base_expansion, demand_gap * 1.5)
    elif strategy.name == "conservative":
        # 保守策略：只满足当前缺口的一部分
        expansion = min(base_expansion, demand_gap * 0.8)
    else:  # moderate
        # 温和策略：满足当前缺口
        expansion = max(base_expansion, demand_gap)
    
    return expansion


if __name__ == "__main__":
    # 测试代码
    print("=== ISRU策略定义测试 ===")
    
    strategies = StrategyDefinitions.get_all_strategies()
    
    for name, strategy in strategies.items():
        print(f"\n{name.upper()} 策略:")
        print(f"  描述: {strategy.description}")
        print(f"  初始部署比例: {strategy.initial_deployment_ratio:.1%}")
        print(f"  利用率阈值: {strategy.utilization_threshold:.1%}")
        print(f"  扩张比例: {strategy.expansion_ratio:.1%}")
        print(f"  风险容忍度: {strategy.risk_tolerance:.1f}")
        print(f"  成本敏感度: {strategy.cost_sensitivity:.1f}")
        
        # 测试决策逻辑
        expected_demand = 1000
        initial_capacity = calculate_initial_capacity(strategy, expected_demand)
        print(f"  初始产能: {initial_capacity:.1f} kg")
        
        # 测试扩张决策
        test_utilization = 0.9
        should_expand = should_expand_capacity(strategy, test_utilization, initial_capacity)
        print(f"  90%利用率时是否扩张: {should_expand}")
        
        if should_expand:
            expansion = calculate_expansion_amount(strategy, initial_capacity, initial_capacity * test_utilization)
            print(f"  扩张数量: {expansion:.1f} kg")