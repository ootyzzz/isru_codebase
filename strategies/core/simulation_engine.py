#!/usr/bin/env python3
"""
策略仿真引擎 - 核心仿真器
基于规则驱动的策略执行，非优化求解
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import json
from dataclasses import asdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.strategy_definitions import StrategyDefinitions, StrategyParams, StrategyType
from strategies.core.state_manager import SystemState, StateManager, Decision
from strategies.core.decision_logic import DecisionEngine
from analysis.gbm_demand import GBMDemandGenerator


class SimulationResult:
    """仿真结果数据结构"""
    
    def __init__(self, strategy_name: str, T: int):
        self.strategy_name = strategy_name
        self.T = T
        self.demand_path: List[float] = []
        self.decisions: List[Decision] = []
        self.states: List[SystemState] = []
        self.performance_metrics: Dict[str, float] = {}
        self.simulation_params: Dict = {}
    
    def to_dict(self) -> Dict:
        """转换为字典格式，便于序列化"""
        return {
            'strategy_name': self.strategy_name,
            'T': self.T,
            'demand_path': self.demand_path,
            'decisions': [asdict(d) for d in self.decisions],
            'states': [asdict(s) for s in self.states],
            'performance_metrics': self.performance_metrics,
            'simulation_params': self.simulation_params
        }


class StrategySimulationEngine:
    """策略仿真引擎"""
    
    def __init__(self, params: Dict):
        """
        初始化仿真引擎
        
        Args:
            params: 系统参数字典
        """
        self.params = params
        self.demand_generator = self._create_demand_generator()
        
    def _create_demand_generator(self) -> GBMDemandGenerator:
        """创建需求生成器"""
        demand_params = self.params['demand']
        return GBMDemandGenerator(
            D0=demand_params['D0'],
            mu=demand_params['mu'],
            sigma=demand_params['sigma'],
            dt=demand_params['dt']
        )
    
    def run_single_simulation(self, 
                            strategy_name: str, 
                            T: int, 
                            seed: Optional[int] = None,
                            demand_path: Optional[List[float]] = None) -> SimulationResult:
        """
        运行单次策略仿真
        
        Args:
            strategy_name: 策略名称
            T: 时间范围
            seed: 随机种子
            demand_path: 预定义的需求路径（可选）
            
        Returns:
            仿真结果
        """
        # 获取策略参数
        strategies = StrategyDefinitions.get_all_strategies(T)
        if strategy_name not in strategies:
            raise ValueError(f"未知策略: {strategy_name}")
        
        strategy = strategies[strategy_name]
        
        # 生成或使用需求路径
        if demand_path is None:
            demand_path = self.demand_generator.generate_single_path(T, seed=seed)
        
        # 确保需求路径长度正确
        if len(demand_path) != T + 1:
            # 调整长度，去掉t=0的初始点
            demand_path = demand_path[1:T+1] if len(demand_path) > T else demand_path[:T]
        
        # 初始化仿真组件
        decision_engine = DecisionEngine(strategy, self.params)
        
        # 新策略不需要在初始化时部署，所有部署都通过决策逻辑处理
        initial_capacity = 0.0
        initial_deployed_mass = 0.0
        
        # 创建初始状态
        initial_state = SystemState(
            current_time=0,
            total_capacity=initial_capacity,
            deployed_mass=initial_deployed_mass,
            current_demand=float(demand_path[0]) if len(demand_path) > 0 else 0.0,
            inventory=0.0
        )
        
        state_manager = StateManager(initial_state)
        
        # 创建结果对象
        result = SimulationResult(strategy_name, T)
        result.demand_path = demand_path
        result.simulation_params = {
            'initial_capacity': initial_capacity,
            'initial_deployed_mass': initial_deployed_mass,
            'strategy_params': asdict(strategy),
            'seed': seed
        }
        
        # 执行仿真循环
        for t in range(T):
            current_demand = float(demand_path[t]) if t < len(demand_path) else 0.0
            
            # 获取需求预测（简单的线性预测）
            demand_forecast = self._generate_demand_forecast(demand_path, t, forecast_horizon=3)
            
            # 做出决策
            decision = decision_engine.make_decision(state_manager.state, demand_forecast)
            
            # 更新状态
            new_state = state_manager.update_state(decision, current_demand, self.params)
            
            # 记录结果
            result.decisions.append(decision)
            result.states.append(SystemState(**asdict(new_state)))  # 创建副本
        
        # 计算性能指标
        result.performance_metrics = state_manager.calculate_performance_metrics(self.params)
        
        return result
    
    def _generate_demand_forecast(self, demand_path: List[float], 
                                current_time: int, 
                                forecast_horizon: int = 3) -> List[float]:
        """
        生成简单的需求预测
        
        Args:
            demand_path: 完整需求路径
            current_time: 当前时间
            forecast_horizon: 预测时间范围
            
        Returns:
            预测的需求序列
        """
        if current_time >= len(demand_path) - 1:
            return []
        
        # 简单线性趋势预测
        available_future = list(demand_path[current_time + 1:current_time + 1 + forecast_horizon])
        
        if len(available_future) < forecast_horizon and current_time > 0:
            # 如果未来数据不足，基于历史趋势预测
            recent_demands = demand_path[max(0, current_time - 2):current_time + 1]
            if len(recent_demands) >= 2:
                growth_rate = (recent_demands[-1] / recent_demands[0]) ** (1 / (len(recent_demands) - 1)) - 1
                last_demand = demand_path[current_time]
                
                # 补充预测值
                for i in range(len(available_future), forecast_horizon):
                    predicted_demand = last_demand * ((1 + growth_rate) ** (i + 1))
                    available_future.append(predicted_demand)
        
        return available_future[:forecast_horizon]
    
    def run_monte_carlo_simulation(self, 
                                 strategy_name: str, 
                                 T: int, 
                                 n_simulations: int = 100,
                                 base_seed: int = 42) -> List[SimulationResult]:
        """
        运行蒙特卡洛仿真
        
        Args:
            strategy_name: 策略名称
            T: 时间范围
            n_simulations: 仿真次数
            base_seed: 基础随机种子
            
        Returns:
            仿真结果列表
        """
        results = []
        
        for i in range(n_simulations):
            seed = base_seed + i
            result = self.run_single_simulation(strategy_name, T, seed=seed)
            results.append(result)
        
        return results
    
    def compare_strategies(self, 
                         strategy_names: List[str], 
                         T: int, 
                         n_simulations: int = 100,
                         base_seed: int = 42) -> Dict[str, List[SimulationResult]]:
        """
        比较多个策略
        
        Args:
            strategy_names: 策略名称列表
            T: 时间范围
            n_simulations: 每个策略的仿真次数
            base_seed: 基础随机种子
            
        Returns:
            策略比较结果
        """
        comparison_results = {}
        
        for strategy_name in strategy_names:
            print(f"正在仿真 {strategy_name} 策略...")
            results = self.run_monte_carlo_simulation(strategy_name, T, n_simulations, base_seed)
            comparison_results[strategy_name] = results
        
        return comparison_results
    
    def calculate_strategy_statistics(self, results: List[SimulationResult]) -> Dict[str, float]:
        """
        计算策略统计信息
        
        Args:
            results: 仿真结果列表
            
        Returns:
            统计信息字典
        """
        if not results:
            return {}
        
        # 提取关键指标
        npvs = [r.performance_metrics.get('npv', 0) for r in results]
        utilizations = [r.performance_metrics.get('avg_utilization', 0) for r in results]
        self_sufficiency_rates = [r.performance_metrics.get('self_sufficiency_rate', 0) for r in results]
        total_costs = [r.performance_metrics.get('total_cost', 0) for r in results]
        
        return {
            # NPV统计
            'npv_mean': float(np.mean(npvs)),
            'npv_std': float(np.std(npvs)),
            'npv_min': float(np.min(npvs)),
            'npv_max': float(np.max(npvs)),
            'npv_p5': float(np.percentile(npvs, 5)),
            'npv_p95': float(np.percentile(npvs, 95)),
            
            # 利用率统计
            'utilization_mean': float(np.mean(utilizations)),
            'utilization_std': float(np.std(utilizations)),
            
            # 自给自足率统计
            'self_sufficiency_mean': float(np.mean(self_sufficiency_rates)),
            'self_sufficiency_std': float(np.std(self_sufficiency_rates)),
            
            # 成本统计
            'total_cost_mean': float(np.mean(total_costs)),
            'total_cost_std': float(np.std(total_costs)),
            
            # 风险指标
            'npv_coefficient_of_variation': float(np.std(npvs) / np.mean(npvs)) if np.mean(npvs) != 0 else 0,
            'probability_positive_npv': float(np.mean([npv > 0 for npv in npvs])),
            
            # 仿真元信息
            'n_simulations': len(results),
            'strategy_name': results[0].strategy_name if results else 'unknown'
        }


if __name__ == "__main__":
    # 测试代码
    print("=== 策略仿真引擎测试 ===")
    
    # 加载参数
    params_path = project_root / "data" / "parameters.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # 创建仿真引擎
    engine = StrategySimulationEngine(params)
    
    # 测试单次仿真
    print("\n--- 单次仿真测试 ---")
    result = engine.run_single_simulation("upfront_deployment", T=10, seed=42)
    print(f"策略: {result.strategy_name}")
    print(f"时间范围: {result.T}")
    print(f"最终NPV: ${result.performance_metrics.get('npv', 0):,.0f}")
    print(f"平均利用率: {result.performance_metrics.get('avg_utilization', 0):.1%}")
    print(f"自给自足率: {result.performance_metrics.get('self_sufficiency_rate', 0):.1%}")
    
    # 测试蒙特卡洛仿真
    print("\n--- 蒙特卡洛仿真测试 ---")
    mc_results = engine.run_monte_carlo_simulation("gradual_deployment", T=10, n_simulations=10, base_seed=42)
    stats = engine.calculate_strategy_statistics(mc_results)
    print(f"NPV均值: ${stats['npv_mean']:,.0f}")
    print(f"NPV标准差: ${stats['npv_std']:,.0f}")
    print(f"平均利用率: {stats['utilization_mean']:.1%}")
    
    # 测试策略比较
    print("\n--- 策略比较测试 ---")
    comparison = engine.compare_strategies(["upfront_deployment", "flexible_deployment"], T=10, n_simulations=5)
    for strategy, results in comparison.items():
        stats = engine.calculate_strategy_statistics(results)
        print(f"{strategy}: NPV=${stats['npv_mean']:,.0f}±${stats['npv_std']:,.0f}")