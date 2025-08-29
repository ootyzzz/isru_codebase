#!/usr/bin/env python3
"""
ISRU策略定义模块 - 生产策略版本
定义三种新的生产策略的参数和决策逻辑
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class StrategyType(Enum):
    """策略类型枚举"""
    UPFRONT_PRODUCTION = "upfront_production"
    GRADUAL_PRODUCTION = "gradual_production"
    FLEXIBLE_PRODUCTION = "flexible_production"


@dataclass
class StrategyParams:
    """策略参数数据结构"""
    name: str
    description: str
    
    # 生产策略类型
    production_type: str             # "upfront", "gradual", "flexible"
    
    # 时间跨度相关参数
    time_horizon: int                # 总时间跨度T
    
    # 生产计算参数
    max_production_ratio: float      # 相对于最终需求的最大生产比例
    
    # 产能建设策略
    capacity_buffer: float           # 产能缓冲比例（确保有足够产能支持生产）
    
    def __post_init__(self):
        """验证参数有效性"""
        assert self.production_type in ["upfront", "gradual", "flexible"], "生产类型必须是upfront, gradual, 或flexible"
        assert self.time_horizon > 0, "时间跨度必须大于0"
        assert 0 <= self.max_production_ratio <= 2.0, "最大生产比例应在0-2.0之间"
        assert 0 <= self.capacity_buffer <= 1.0, "产能缓冲比例应在0-1.0之间"


class StrategyDefinitions:
    """策略定义类 - 生产策略版本"""
    
    @staticmethod
    def get_upfront_production_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        一次性满产策略：第一年就生产量拉满到最终需求水平
        - 第一年就达到最终需求的生产水平
        - 需要提前建设足够的产能支持满产
        - 后续年份维持高生产水平
        """
        return StrategyParams(
            name="upfront_production",
            description="Upfront Production: Reach maximum production in the first year",
            production_type="upfront",
            time_horizon=time_horizon,
            max_production_ratio=1.0,       # 生产100%的最终需求
            capacity_buffer=0.2             # 20%的产能缓冲
        )
    
    @staticmethod
    def get_gradual_production_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        渐进生产策略：生产量逐年平均增长
        - 生产量从0逐步增长到最终需求水平
        - 每年生产量 = (年份/T) * 最终需求
        - 产能建设跟随生产计划
        """
        return StrategyParams(
            name="gradual_production",
            description="Gradual Production: Gradually increase production to final demand level",
            production_type="gradual",
            time_horizon=time_horizon,
            max_production_ratio=1.0,       # 生产100%的最终需求
            capacity_buffer=0.1             # 10%的产能缓冲
        )
    
    @staticmethod
    def get_flexible_production_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        灵活生产策略：根据当年实际需求动态调整生产量
        - 每年根据当年的实际需求调整生产量
        - 生产量 = min(当年需求, 当前产能)
        - 产能建设根据需求预测进行
        """
        return StrategyParams(
            name="flexible_production",
            description="Flexible Production: Adjust production based on actual annual demand",
            production_type="flexible",
            time_horizon=time_horizon,
            max_production_ratio=1.0,       # 生产100%的当年需求
            capacity_buffer=0.15            # 15%的产能缓冲
        )
    
    @classmethod
    def get_strategy(cls, strategy_type: StrategyType, time_horizon: int = 20) -> StrategyParams:
        """根据策略类型获取策略参数"""
        strategy_map = {
            StrategyType.UPFRONT_PRODUCTION: lambda: cls.get_upfront_production_strategy(time_horizon),
            StrategyType.GRADUAL_PRODUCTION: lambda: cls.get_gradual_production_strategy(time_horizon),
            StrategyType.FLEXIBLE_PRODUCTION: lambda: cls.get_flexible_production_strategy(time_horizon)
        }
        
        if strategy_type not in strategy_map:
            raise ValueError(f"未知的策略类型: {strategy_type}")
        
        return strategy_map[strategy_type]()
    
    @classmethod
    def get_all_strategies(cls, time_horizon: int = 20) -> Dict[str, StrategyParams]:
        """获取所有策略"""
        return {
            "upfront_production": cls.get_upfront_production_strategy(time_horizon),
            "gradual_production": cls.get_gradual_production_strategy(time_horizon),
            "flexible_production": cls.get_flexible_production_strategy(time_horizon)
        }


def calculate_production_for_year(strategy: StrategyParams, 
                                year: int,
                                final_demand: float,
                                current_demand: float = 0) -> float:
    """
    根据策略计算某年的生产量
    
    Args:
        strategy: 策略参数
        year: 当前年份 (0-based)
        final_demand: 最终年份的预期需求
        current_demand: 当年的实际需求
        
    Returns:
        当年的目标生产量
    """
    max_production = final_demand * strategy.max_production_ratio
    
    if strategy.production_type == "upfront":
        # 一次性满产：从第一年就达到最大生产量
        return max_production
        
    elif strategy.production_type == "gradual":
        # 渐进生产：生产量逐年线性增长
        progress = (year + 1) / strategy.time_horizon  # +1 因为year是0-based
        return max_production * progress
        
    elif strategy.production_type == "flexible":
        # 灵活生产：根据当年实际需求调整
        return min(current_demand, max_production)
    
    return 0


def calculate_required_capacity(strategy: StrategyParams,
                              target_production: float) -> float:
    """
    根据目标生产量计算所需产能
    
    Args:
        strategy: 策略参数
        target_production: 目标生产量
        
    Returns:
        所需产能（包含缓冲）
    """
    return target_production * (1 + strategy.capacity_buffer)


def should_expand_capacity(strategy: StrategyParams, 
                          current_utilization: float,
                          current_capacity: float,
                          demand_trend: Optional[List[float]] = None) -> bool:
    """
    判断是否应该扩张产能 - 生产策略版本
    
    Args:
        strategy: 策略参数
        current_utilization: 当前利用率
        current_capacity: 当前产能
        demand_trend: 需求趋势（可选）
        
    Returns:
        是否应该扩张
    """
    # 生产策略基于产能需求而非利用率触发扩张
    return False


def calculate_expansion_amount(strategy: StrategyParams,
                             current_capacity: float,
                             current_demand: float) -> float:
    """
    计算扩张数量 - 生产策略版本
    
    Args:
        strategy: 策略参数
        current_capacity: 当前产能
        current_demand: 当前需求
        
    Returns:
        扩张数量
    """
    # 生产策略的扩张量由生产计划决定，这里返回0
    return 0


if __name__ == "__main__":
    # 测试代码
    print("=== ISRU生产策略定义测试 ===")
    
    time_horizon = 20
    strategies = StrategyDefinitions.get_all_strategies(time_horizon)
    
    for name, strategy in strategies.items():
        print(f"\n{name.upper()} 策略:")
        print(f"  描述: {strategy.description}")
        print(f"  生产类型: {strategy.production_type}")
        print(f"  时间跨度: {strategy.time_horizon}")
        print(f"  最大生产比例: {strategy.max_production_ratio:.1%}")
        print(f"  产能缓冲: {strategy.capacity_buffer:.1%}")
        
        # 测试生产逻辑
        final_demand = 1000
        print(f"  生产测试（最终需求: {final_demand} kg）:")
        
        for year in range(min(5, time_horizon)):
            current_demand = final_demand * (year + 1) / time_horizon  # 模拟需求增长
            production = calculate_production_for_year(
                strategy, year, final_demand, current_demand
            )
            required_capacity = calculate_required_capacity(strategy, production)
            print(f"    第{year+1}年: 生产{production:.1f}kg, 需要产能{required_capacity:.1f}kg")