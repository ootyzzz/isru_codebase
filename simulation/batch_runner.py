"""
批量仿真运行器
支持多场景、多参数组合的批量仿真
"""

import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import pickle

from models.isru_model import ISRUOptimizationModel
from analysis.gbm_demand import GBMDemandGenerator

logger = logging.getLogger(__name__)


class BatchRunner:
    """
    批量仿真运行器
    
    支持以下功能：
    - 多场景批量仿真
    - 并行计算
    - 结果聚合和分析
    - 进度跟踪
    """
    
    def __init__(self, params: Dict[str, Any], output_dir: str = "results"):
        """
        初始化批量运行器
        
        Args:
            params: 基础参数字典
            output_dir: 输出目录
        """
        self.params = params
        self.output_dir = output_dir
        self.results = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """设置日志配置"""
        log_file = os.path.join(self.output_dir, f"batch_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_single_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行单个场景
        
        Args:
            scenario: 场景配置
            
        Returns:
            场景结果
        """
        logger.info(f"运行场景: {scenario.get('name', 'unnamed')}")
        
        try:
            # 合并参数
            scenario_params = self._merge_params(self.params, scenario.get('params', {}))
            
            # 生成需求路径
            demand_generator = GBMDemandGenerator(scenario_params)
            demand_path = demand_generator.generate_path(
                n_scenarios=1,
                random_seed=scenario.get('seed', 42)
            )[0]
            
            # 创建并求解模型
            model = ISRUOptimizationModel(scenario_params)
            model.build_model(demand_path)
            result = model.solve()
            
            # 添加场景信息
            result['scenario'] = scenario
            result['demand_path'] = demand_path
            
            return result
            
        except Exception as e:
            logger.error(f"场景运行失败: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'scenario': scenario
            }
    
    def run_batch(self, scenarios: List[Dict[str, Any]], 
                  parallel: bool = True, max_workers: int = None) -> List[Dict[str, Any]]:
        """
        运行批量仿真
        
        Args:
            scenarios: 场景列表
            parallel: 是否并行运行
            max_workers: 最大工作进程数
            
        Returns:
            所有场景的结果列表
        """
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(scenarios))
        
        logger.info(f"开始批量仿真，共{len(scenarios)}个场景")
        
        if parallel and len(scenarios) > 1:
            # 并行运行
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_scenario = {
                    executor.submit(self.run_single_scenario, scenario): scenario 
                    for scenario in scenarios
                }
                
                for future in as_completed(future_to_scenario):
                    result = future.result()
                    self.results.append(result)
                    
                    # 保存中间结果
                    self._save_intermediate_result(result)
                    
        else:
            # 串行运行
            for scenario in scenarios:
                result = self.run_single_scenario(scenario)
                self.results.append(result)
                
                # 保存中间结果
                self._save_intermediate_result(result)
        
        logger.info("批量仿真完成")
        
        # 保存完整结果
        self._save_all_results()
        
        return self.results
    
    def _merge_params(self, base_params: Dict, override_params: Dict) -> Dict:
        """合并参数字典"""
        merged = base_params.copy()
        
        def deep_merge(d1, d2):
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value
        
        deep_merge(merged, override_params)
        return merged
    
    def _save_intermediate_result(self, result: Dict[str, Any]) -> None:
        """保存中间结果"""
        scenario_name = result['scenario'].get('name', 'unnamed')
        filename = f"result_{scenario_name}_{datetime.now().strftime('%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 转换为可序列化的格式
        serializable_result = self._make_serializable(result)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
    
    def _save_all_results(self) -> None:
        """保存所有结果"""
        filename = f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"所有结果已保存到: {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def get_results_summary(self) -> pd.DataFrame:
        """获取结果摘要"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for result in self.results:
            if result['status'] == 'optimal':
                summary = {
                    'scenario': result['scenario'].get('name', 'unnamed'),
                    'status': result['status'],
                    'objective_value': result['objective_value'],
                    'total_demand': sum(result['demand_path']),
                    'total_supply': sum(result['solution']['Qt'].values()) if 'solution' in result else 0,
                    'total_shortage': sum(result['solution']['St'].values()) if 'solution' in result else 0,
                    'max_capacity': max(result['solution']['Qt_cap'].values()) if 'solution' in result else 0,
                    'total_mass': max(result['solution']['Mt'].values()) if 'solution' in result else 0
                }
                
                # 添加场景参数
                for key, value in result['scenario'].get('params', {}).items():
                    if isinstance(value, (int, float, str)):
                        summary[f'param_{key}'] = value
                
                summary_data.append(summary)
            else:
                summary = {
                    'scenario': result['scenario'].get('name', 'unnamed'),
                    'status': result['status'],
                    'error': result.get('error', '')
                }
                summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_summary_report(self) -> str:
        """保存摘要报告"""
        summary_df = self.get_results_summary()
        
        if summary_df.empty:
            logger.warning("没有可保存的结果")
            return ""
        
        filename = f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        summary_df.to_csv(filepath, index=False)
        logger.info(f"摘要报告已保存到: {filepath}")
        
        return filepath


class ScenarioGenerator:
    """场景生成器"""
    
    @staticmethod
    def generate_parameter_sweep(base_params: Dict, 
                                 param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        生成参数扫描场景
        
        Args:
            base_params: 基础参数
            param_ranges: 参数范围字典
            
        Returns:
            场景列表
        """
        scenarios = []
        
        # 生成参数组合
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        
        from itertools import product
        
        for i, combination in enumerate(product(*param_values)):
            scenario = {
                'name': f'param_sweep_{i}',
                'params': {}
            }
            
            # 构建参数层次结构
            for name, value in zip(param_names, combination):
                keys = name.split('.')
                current = scenario['params']
                
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                current[keys[-1]] = value
            
            scenarios.append(scenario)
        
        return scenarios
    
    @staticmethod
    def generate_demand_scenarios(base_params: Dict, 
                                  demand_configs: List[Dict]) -> List[Dict[str, Any]]:
        """
        生成需求场景
        
        Args:
            base_params: 基础参数
            demand_configs: 需求配置列表
            
        Returns:
            场景列表
        """
        scenarios = []
        
        for i, config in enumerate(demand_configs):
            scenario = {
                'name': f'demand_scenario_{i}',
                'params': {
                    'demand': config
                },
                'seed': config.get('seed', 42 + i)
            }
            scenarios.append(scenario)
        
        return scenarios
    
    @staticmethod
    def generate_uncertainty_scenarios(base_params: Dict, 
                                       uncertainty_factors: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        生成不确定性场景
        
        Args:
            base_params: 基础参数
            uncertainty_factors: 不确定性因子
            
        Returns:
            场景列表
        """
        scenarios = []
        
        # 生成不确定性组合
        factor_names = list(uncertainty_factors.keys())
        factor_values = [uncertainty_factors[name] for name in factor_names]
        
        from itertools import product
        
        for i, combination in enumerate(product(*factor_values)):
            scenario = {
                'name': f'uncertainty_{i}',
                'params': {
                    'costs': {},
                    'technology': {},
                    'economics': {}
                }
            }
            
            # 应用不确定性因子
            for name, value in zip(factor_names, combination):
                if name.startswith('cost_'):
                    scenario['params']['costs'][name[5:]] = base_params['costs'][name[5:]] * value
                elif name.startswith('tech_'):
                    scenario['params']['technology'][name[5:]] = base_params['technology'][name[5:]] * value
                elif name.startswith('econ_'):
                    scenario['params']['economics'][name[5:]] = base_params['economics'][name[5:]] * value
            
            scenarios.append(scenario)
        
        return scenarios


# 便捷函数
def run_parameter_sweep(params: Dict, param_ranges: Dict, 
                        output_dir: str = "results") -> pd.DataFrame:
    """
    运行参数扫描
    
    Args:
        params: 基础参数
        param_ranges: 参数范围
        output_dir: 输出目录
        
    Returns:
        结果摘要DataFrame
    """
    generator = ScenarioGenerator()
    scenarios = generator.generate_parameter_sweep(params, param_ranges)
    
    runner = BatchRunner(params, output_dir)
    results = runner.run_batch(scenarios)
    
    return runner.get_results_summary()


if __name__ == "__main__":
    # 测试批量运行器
    import json
    
    # 加载参数
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    
    # 创建测试场景
    scenarios = [
        {
            'name': 'baseline',
            'params': {}
        },
        {
            'name': 'high_demand',
            'params': {
                'demand': {'D0': params['demand']['D0'] * 1.5}
            }
        },
        {
            'name': 'low_cost',
            'params': {
                'costs': {'c_op': params['costs']['c_op'] * 0.8}
            }
        }
    ]
    
    # 运行批量仿真
    runner = BatchRunner(params)
    results = runner.run_batch(scenarios, parallel=False)
    
    # 保存摘要
    summary_file = runner.save_summary_report()
    print(f"结果摘要已保存到: {summary_file}")