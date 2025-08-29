#!/usr/bin/env python3
"""
ISRU策略定义模块 - 新策略版本
定义三种新的部署策略的参数和决策逻辑
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class StrategyType(Enum):
    """策略类型枚举"""
    UPFRONT_DEPLOYMENT = "upfront_deployment"
    GRADUAL_DEPLOYMENT = "gradual_deployment"
    FLEXIBLE_DEPLOYMENT = "flexible_deployment"


@dataclass
class StrategyParams:
    """策略参数数据结构"""
    name: str
    description: str
    
    # 部署策略类型
    deployment_type: str             # "upfront", "gradual", "flexible"
    
    # 时间跨度相关参数
    time_horizon: int                # 总时间跨度T
    
    # 部署计算参数
    total_deployment_ratio: float    # 相对于最终需求的总部署比例
    
    # 灵活策略特有参数
    response_threshold: float        # 响应阈值（供需差异比例）
    
    def __post_init__(self):
        """验证参数有效性"""
        assert self.deployment_type in ["upfront", "gradual", "flexible"], "部署类型必须是upfront, gradual, 或flexible"
        assert self.time_horizon > 0, "时间跨度必须大于0"
        assert 0 <= self.total_deployment_ratio <= 2.0, "总部署比例应在0-2.0之间"
        assert 0 <= self.response_threshold <= 1.0, "响应阈值应在0-1.0之间"


class StrategyDefinitions:
    """策略定义类 - 新策略版本"""
    
    @staticmethod
    def get_upfront_deployment_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        一次性全部部署策略：第一年部署最终的总部署量
        - 前期大量投资，一次性部署到位
        - 后续年份不再新增部署，仅维持运营
        - 基于T年的需求预期确定总部署量
        """
        return StrategyParams(
            name="upfront_deployment",
            description="Upfront Deployment: Deploy all capacity in the first year",
            deployment_type="upfront",
            time_horizon=time_horizon,
            total_deployment_ratio=1.0,     # 部署100%的最终需求
            response_threshold=0.0          # 不适用
        )
    
    @staticmethod
    def get_gradual_deployment_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        平均分布部署策略：将总部署量平均分配到每一年
        - 均匀分布投资，稳步扩张
        - 每年新增量 = 最终总部署量 / T
        - 系统开始时就知道时间跨度T
        """
        return StrategyParams(
            name="gradual_deployment",
            description="Gradual Deployment: Distribute total capacity evenly across all years",
            deployment_type="gradual",
            time_horizon=time_horizon,
            total_deployment_ratio=1.0,     # 部署100%的最终需求
            response_threshold=0.0          # 不适用
        )
    
    @staticmethod
    def get_flexible_deployment_strategy(time_horizon: int = 20) -> StrategyParams:
        """
        灵活部署策略：根据实际供需情况动态调整
        - 响应式部署，根据供需差异动态调整
        - 如果第t年供应量 < 需求量，则第t+1年新增部署差额
        - 刻舟求剑式策略，总是试图弥补上一年的缺口
        """
        return StrategyParams(
            name="flexible_deployment",
            description="Flexible Deployment: Respond to supply-demand gaps dynamically",
            deployment_type="flexible",
            time_horizon=time_horizon,
            total_deployment_ratio=1.0,     # 理论上的最大部署比例
            response_threshold=0.05         # 5%的响应阈值
        )
    
    @classmethod
    def get_strategy(cls, strategy_type: StrategyType, time_horizon: int = 20) -> StrategyParams:
        """根据策略类型获取策略参数"""
        strategy_map = {
            StrategyType.UPFRONT_DEPLOYMENT: lambda: cls.get_upfront_deployment_strategy(time_horizon),
            StrategyType.GRADUAL_DEPLOYMENT: lambda: cls.get_gradual_deployment_strategy(time_horizon),
            StrategyType.FLEXIBLE_DEPLOYMENT: lambda: cls.get_flexible_deployment_strategy(time_horizon)
        }
        
        if strategy_type not in strategy_map:
            raise ValueError(f"未知的策略类型: {strategy_type}")
        
        return strategy_map[strategy_type]()
    
    @classmethod
    def get_all_strategies(cls, time_horizon: int = 20) -> Dict[str, StrategyParams]:
        """获取所有策略"""
        return {
            "upfront_deployment": cls.get_upfront_deployment_strategy(time_horizon),
            "gradual_deployment": cls.get_gradual_deployment_strategy(time_horizon),
            "flexible_deployment": cls.get_flexible_deployment_strategy(time_horizon)
        }


def calculate_deployment_for_year(strategy: StrategyParams,
                                year: int,
                                final_demand: float,
                                previous_supply: float = 0,
                                previous_demand: float = 0) -> float:
    """
    根据策略计算某年的部署量
    
    Args:
        strategy: 策略参数
        year: 当前年份 (0-based)
        final_demand: 最终年份的预期需求
        previous_supply: 上一年的总供应量（仅灵活策略使用）
        previous_demand: 上一年的需求量（仅灵活策略使用）
        
    Returns:
        当年的新增部署量
    """
    total_capacity_needed = final_demand * strategy.total_deployment_ratio
    
    print(f"[STRATEGY DEBUG] {strategy.deployment_type} - Year {year}")
    print(f"[STRATEGY DEBUG] Final demand: {final_demand:.1f}, Total capacity needed: {total_capacity_needed:.1f}")
    
    if strategy.deployment_type == "upfront":
        # 一次性全部部署：第一年部署全部，其他年份为0
        deployment = total_capacity_needed if year == 0 else 0
        print(f"[STRATEGY DEBUG] Upfront deployment: {deployment:.1f}")
        return deployment
        
    elif strategy.deployment_type == "gradual":
        # 平均分布部署：每年部署相同数量
        deployment = total_capacity_needed / strategy.time_horizon
        print(f"[STRATEGY DEBUG] Gradual deployment: {deployment:.1f} (total/{strategy.time_horizon})")
        return deployment
        
    elif strategy.deployment_type == "flexible":
        # 灵活部署：根据上一年的供需差异决定
        if year == 0:
            # 第一年部署一个基础量（比如20%的最终需求）
            deployment = total_capacity_needed * 0.2
            print(f"[STRATEGY DEBUG] Flexible deployment (Year 0): {deployment:.1f}")
            return deployment
        else:
            # 根据上一年的供需差异决定
            print(f"[STRATEGY DEBUG] Previous supply: {previous_supply:.1f}, Previous demand: {previous_demand:.1f}")
            if previous_demand > previous_supply:
                gap = previous_demand - previous_supply
                gap_ratio = gap / previous_demand if previous_demand > 0 else 0
                print(f"[STRATEGY DEBUG] Gap: {gap:.1f}, Gap ratio: {gap_ratio:.3f}, Threshold: {strategy.response_threshold:.3f}")
                # 如果差异超过阈值，则部署差额
                if gap_ratio > strategy.response_threshold:
                    print(f"[STRATEGY DEBUG] Flexible deployment (Gap response): {gap:.1f}")
                    return gap
            print(f"[STRATEGY DEBUG] Flexible deployment: 0 (no gap or below threshold)")
            return 0
    
    return 0


def should_expand_capacity(strategy: StrategyParams, 
                          current_utilization: float,
                          current_capacity: float,
                          demand_trend: Optional[List[float]] = None) -> bool:
    """
    判断是否应该扩张产能 - 新策略版本
    
    Args:
        strategy: 策略参数
        current_utilization: 当前利用率
        current_capacity: 当前产能
        demand_trend: 需求趋势（可选）
        
    Returns:
        是否应该扩张
    """
    # 新策略不使用传统的利用率触发扩张
    # 扩张决策完全由策略的部署逻辑决定
    return False


def calculate_expansion_amount(strategy: StrategyParams,
                             current_capacity: float,
                             current_demand: float) -> float:
    """
    计算扩张数量 - 新策略版本
    
    Args:
        strategy: 策略参数
        current_capacity: 当前产能
        current_demand: 当前需求
        
    Returns:
        扩张数量
    """
    # 新策略的扩张量由部署逻辑决定，这里返回0
    return 0


if __name__ == "__main__":
    # 测试代码
    print("=== ISRU新策略定义测试 ===")
    
    time_horizon = 20
    strategies = StrategyDefinitions.get_all_strategies(time_horizon)
    
    for name, strategy in strategies.items():
        print(f"\n{name.upper()} 策略:")
        print(f"  描述: {strategy.description}")
        print(f"  部署类型: {strategy.deployment_type}")
        print(f"  时间跨度: {strategy.time_horizon}")
        print(f"  总部署比例: {strategy.total_deployment_ratio:.1%}")
        print(f"  响应阈值: {strategy.response_threshold:.1%}")
        
        # 测试部署逻辑
        final_demand = 1000
        print(f"  部署测试（最终需求: {final_demand} kg）:")
        
        for year in range(min(5, time_horizon)):
            deployment = calculate_deployment_for_year(
                strategy, year, final_demand, 
                previous_supply=800 if year > 0 else 0,
                previous_demand=900 if year > 0 else 0
            )
            print(f"    第{year+1}年部署: {deployment:.1f} kg")