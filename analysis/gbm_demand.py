"""
Geometric Brownian Motion (GBM) Demand Generator
用于生成月球氧气需求的蒙特卡洛仿真路径
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GBMDemandGenerator:
    """
    几何布朗运动需求生成器
    
    实现公式: D_t = D_{t-1} * exp((μ - 0.5σ²)dt + σ√dt * Z_t)
    其中 Z_t ~ N(0,1) 是标准正态随机变量
    """
    
    def __init__(self, D0: float, mu: float, sigma: float, dt: float = 1.0):
        """
        初始化GBM生成器
        
        Args:
            D0: 初始需求
            mu: 漂移率 (drift)
            sigma: 波动率 (volatility)
            dt: 时间步长
        """
        self.D0 = D0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        
    def generate_single_path(self, T: int, seed: Optional[int] = None) -> np.ndarray:
        """
        生成单条需求路径
        
        Args:
            T: 时间范围（年数）
            seed: 随机种子
            
        Returns:
            需求路径数组，形状为 (T+1,)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # 初始化路径
        path = np.zeros(T + 1)
        path[0] = self.D0
        
        # GBM参数
        drift = (self.mu - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        # 生成路径
        for t in range(1, T + 1):
            z = np.random.standard_normal()
            path[t] = path[t-1] * np.exp(drift + diffusion * z)
            
        return path
    
    def generate_multiple_paths(self, T: int, n_scenarios: int, 
                               seed: Optional[int] = None) -> pd.DataFrame:
        """
        生成多条需求路径
        
        Args:
            T: 时间范围（年数）
            n_scenarios: 场景数量
            seed: 随机种子
            
        Returns:
            DataFrame，每列是一条需求路径
        """
        if seed is not None:
            np.random.seed(seed)
            
        paths = []
        for scenario in range(n_scenarios):
            path = self.generate_single_path(T)
            paths.append(path)
            
        # 创建DataFrame
        df = pd.DataFrame(
            np.array(paths).T,
            columns=[f'scenario_{i+1}' for i in range(n_scenarios)],
            index=range(T + 1)
        )
        
        return df
    
    def generate_scenarios_with_stats(self, T: int, n_scenarios: int, 
                                    seed: Optional[int] = None) -> dict:
        """
        生成需求场景并计算统计量
        
        Returns:
            包含路径和统计量的字典
        """
        df = self.generate_multiple_paths(T, n_scenarios, seed)
        
        stats = {
            'paths': df,
            'mean': df.mean(axis=1),
            'std': df.std(axis=1),
            'percentile_5': df.quantile(0.05, axis=1),
            'percentile_95': df.quantile(0.95, axis=1),
            'min': df.min(axis=1),
            'max': df.max(axis=1)
        }
        
        return stats
    
    def save_scenarios(self, df: pd.DataFrame, filepath: str) -> None:
        """保存需求场景到CSV文件"""
        df.to_csv(filepath)
        logger.info(f"需求场景已保存到: {filepath}")
        
    def load_scenarios(self, filepath: str) -> pd.DataFrame:
        """从CSV文件加载需求场景"""
        df = pd.read_csv(filepath, index_col=0)
        logger.info(f"需求场景已从 {filepath} 加载")
        return df


def create_demand_generator_from_params(params: dict) -> GBMDemandGenerator:
    """
    从参数字典创建GBM生成器
    
    Args:
        params: 包含需求参数的参数字典
        
    Returns:
        GBMDemandGenerator实例
    """
    demand_params = params['demand']
    return GBMDemandGenerator(
        D0=demand_params['D0'],
        mu=demand_params['mu'],
        sigma=demand_params['sigma'],
        dt=demand_params['dt']
    )


if __name__ == "__main__":
    # 测试代码
    import json
    
    # 加载参数
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    
    # 创建生成器
    generator = create_demand_generator_from_params(params)
    
    # 生成场景
    T = params['economics']['T']
    n_scenarios = params['simulation']['n_scenarios']
    seed = params['simulation']['random_seed']
    
    stats = generator.generate_scenarios_with_stats(T, n_scenarios, seed)
    
    # 保存结果
    generator.save_scenarios(stats['paths'], 'data/demand_scenarios.csv')
    
    # 打印统计信息
    print("需求场景统计信息:")
    print(stats['mean'])
    print(f"\n场景数量: {n_scenarios}")
    print(f"时间范围: {T} 年")