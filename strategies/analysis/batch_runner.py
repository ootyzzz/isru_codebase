#!/usr/bin/env python3
"""
批量仿真执行器
支持多策略、多时间跨度的批量仿真和对比分析
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.simulation_engine import StrategySimulationEngine, SimulationResult
from strategies.core.strategy_definitions import StrategyDefinitions
from strategies.utils.terminal_display import TerminalDisplay
from strategies.utils.result_manager import ResultManager


class BatchSimulationRunner:
    """批量仿真执行器"""
    
    def __init__(self, params: Dict, results_dir: Optional[Path] = None):
        """
        初始化批量仿真执行器
        
        Args:
            params: 系统参数
            results_dir: 结果保存目录
        """
        self.params = params
        self.results_dir = results_dir or Path(__file__).parent.parent / "simulation_results"
        
        # 创建仿真引擎和结果管理器
        self.engine = StrategySimulationEngine(params)
        self.result_manager = ResultManager(self.results_dir)
        
        # 默认配置
        self.default_strategies = ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
        self.default_time_horizons = [10, 20, 30, 40, 50]
        self.default_n_simulations = 100
    
    def run_single_batch(self, 
                        strategy_name: str, 
                        T: int, 
                        n_simulations: int = 100,
                        base_seed: int = 42,
                        save_results: bool = True,
                        show_progress: bool = True) -> Tuple[List[SimulationResult], Dict[str, float]]:
        """
        运行单个策略的批量仿真
        
        Args:
            strategy_name: 策略名称
            T: 时间范围
            n_simulations: 仿真次数
            base_seed: 基础随机种子
            save_results: 是否保存结果
            show_progress: 是否显示进度
            
        Returns:
            (仿真结果列表, 统计信息)
        """
        if show_progress:
            TerminalDisplay.print_section(f"运行 {strategy_name.title()} 策略 (T={T})")
        
        results = []
        
        # 运行仿真
        for i in range(n_simulations):
            if show_progress and i % 10 == 0:
                TerminalDisplay.print_simulation_status(strategy_name, T, i, n_simulations)
            
            seed = base_seed + i
            result = self.engine.run_single_simulation(strategy_name, T, seed=seed)
            results.append(result)
        
        if show_progress:
            TerminalDisplay.print_simulation_status(strategy_name, T, n_simulations, n_simulations)
        
        # 计算统计信息
        stats = self.engine.calculate_strategy_statistics(results)
        
        # 保存结果
        if save_results:
            metadata = {
                'n_simulations': n_simulations,
                'base_seed': base_seed,
                'simulation_type': 'single_batch'
            }
            saved_files = self.result_manager.save_simulation_batch(
                strategy_name, T, results, stats, metadata
            )
            if show_progress:
                print(f"结果已保存: {len(saved_files)} 个文件")
        
        return results, stats
    
    def run_strategy_comparison(self, 
                              strategies: Optional[List[str]] = None,
                              T: int = 30,
                              n_simulations: int = 100,
                              base_seed: int = 42,
                              save_results: bool = True,
                              show_progress: bool = True) -> Dict[str, Tuple[List[SimulationResult], Dict[str, float]]]:
        """
        运行策略对比仿真
        
        Args:
            strategies: 策略列表
            T: 时间范围
            n_simulations: 每个策略的仿真次数
            base_seed: 基础随机种子
            save_results: 是否保存结果
            show_progress: 是否显示进度
            
        Returns:
            策略对比结果
        """
        if strategies is None:
            strategies = self.default_strategies
        
        if show_progress:
            TerminalDisplay.print_header(f"策略对比仿真 (T={T}年)", width=70)
        
        comparison_results = {}
        
        for strategy in strategies:
            results, stats = self.run_single_batch(
                strategy, T, n_simulations, base_seed, save_results, show_progress
            )
            comparison_results[strategy] = (results, stats)
        
        # 显示对比结果
        if show_progress:
            self._display_comparison_results(comparison_results, T)
        
        # 保存对比结果
        if save_results:
            self.result_manager.save_comparison_results(comparison_results, T, "strategy_comparison")
        
        return comparison_results
    
    def run_time_horizon_analysis(self, 
                                 time_horizons: Optional[List[int]] = None,
                                 strategies: Optional[List[str]] = None,
                                 n_simulations: int = 100,
                                 base_seed: int = 42,
                                 save_results: bool = True,
                                 show_progress: bool = True) -> Dict[int, Dict[str, Tuple[List[SimulationResult], Dict[str, float]]]]:
        """
        运行时间跨度分析
        
        Args:
            time_horizons: 时间跨度列表
            strategies: 策略列表
            n_simulations: 每个策略的仿真次数
            base_seed: 基础随机种子
            save_results: 是否保存结果
            show_progress: 是否显示进度
            
        Returns:
            时间跨度分析结果
        """
        if time_horizons is None:
            time_horizons = self.default_time_horizons
        if strategies is None:
            strategies = self.default_strategies
        
        if show_progress:
            TerminalDisplay.print_header("时间跨度分析", width=70)
        
        horizon_results = {}
        
        for T in time_horizons:
            if show_progress:
                TerminalDisplay.print_section(f"时间跨度: {T}年")
            
            comparison = self.run_strategy_comparison(
                strategies, T, n_simulations, base_seed, save_results, False
            )
            horizon_results[T] = comparison
        
        # 显示时间跨度分析结果
        if show_progress:
            self._display_horizon_analysis(horizon_results)
        
        # 保存时间跨度分析结果
        if save_results:
            self.result_manager.save_horizon_analysis(horizon_results, strategies)
        
        return horizon_results
    
    def run_parallel_batch(self, 
                          strategies: Optional[List[str]] = None,
                          time_horizons: Optional[List[int]] = None,
                          n_simulations: int = 100,
                          base_seed: int = 42,
                          max_workers: Optional[int] = None,
                          save_results: bool = True) -> Dict[int, Dict[str, Tuple[List[SimulationResult], Dict[str, float]]]]:
        """
        并行运行批量仿真
        
        Args:
            strategies: 策略列表
            time_horizons: 时间跨度列表
            n_simulations: 每个策略的仿真次数
            base_seed: 基础随机种子
            max_workers: 最大并行工作进程数
            save_results: 是否保存结果
            
        Returns:
            并行仿真结果
        """
        if strategies is None:
            strategies = self.default_strategies
        if time_horizons is None:
            time_horizons = self.default_time_horizons
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(strategies) * len(time_horizons))
        
        TerminalDisplay.print_header("并行批量仿真", width=70)
        print(f"策略: {', '.join(strategies)}")
        print(f"时间跨度: {time_horizons}")
        print(f"仿真次数: {n_simulations}/策略")
        print(f"并行进程: {max_workers}")
        print()
        
        # 准备任务列表
        tasks = []
        for T in time_horizons:
            for strategy in strategies:
                tasks.append((strategy, T, n_simulations, base_seed))
        
        # 并行执行
        results = {}
        completed_tasks = 0
        total_tasks = len(tasks)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._run_single_task, task): task 
                for task in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                strategy, T, _, _ = task
                
                try:
                    task_results, task_stats = future.result()
                    
                    # 组织结果
                    if T not in results:
                        results[T] = {}
                    results[T][strategy] = (task_results, task_stats)
                    
                    # 保存结果
                    if save_results:
                        metadata = {
                            'n_simulations': n_simulations,
                            'base_seed': base_seed,
                            'simulation_type': 'parallel_batch'
                        }
                        self.result_manager.save_simulation_batch(
                            strategy, T, task_results, task_stats, metadata
                        )
                    
                    completed_tasks += 1
                    TerminalDisplay.print_progress_bar(
                        completed_tasks, total_tasks, 
                        prefix="总进度", 
                        suffix=f"{completed_tasks}/{total_tasks} 任务完成"
                    )
                    
                except Exception as e:
                    print(f"任务 {task} 执行失败: {e}")
        
        print("\n并行仿真完成！")
        return results
    
    def _run_single_task(self, task: Tuple[str, int, int, int]) -> Tuple[List[SimulationResult], Dict[str, float]]:
        """运行单个仿真任务（用于并行执行）"""
        strategy, T, n_simulations, base_seed = task
        
        # 在子进程中创建新的引擎实例
        engine = StrategySimulationEngine(self.params)
        results = []
        
        for i in range(n_simulations):
            seed = base_seed + i
            result = engine.run_single_simulation(strategy, T, seed=seed)
            results.append(result)
        
        stats = engine.calculate_strategy_statistics(results)
        return results, stats
    
    def export_results_to_excel(self,
                               strategies: Optional[List[str]] = None,
                               time_horizons: Optional[List[int]] = None,
                               output_file: Optional[Path] = None) -> Path:
        """
        导出结果到Excel文件
        
        Args:
            strategies: 策略列表
            time_horizons: 时间跨度列表
            output_file: 输出文件路径
            
        Returns:
            Excel文件路径
        """
        if strategies is None:
            strategies = self.default_strategies
        if time_horizons is None:
            time_horizons = self.default_time_horizons
        
        return self.result_manager.export_to_excel(strategies, time_horizons, output_file)
    
    def get_available_results(self) -> Dict[str, List[str]]:
        """获取可用的结果列表"""
        return self.result_manager.get_available_results()
    
    def load_previous_results(self, strategy_name: str, T: int) -> Optional[Dict]:
        """加载之前的仿真结果"""
        return self.result_manager.load_simulation_results(strategy_name, T)
    
    def cleanup_old_results(self, keep_days: int = 30):
        """清理旧的结果文件"""
        self.result_manager.cleanup_old_results(keep_days)
    
    def _display_comparison_results(self, comparison_results: Dict, T: int):
        """显示策略对比结果"""
        TerminalDisplay.print_section(f"T={T}年 策略对比结果")
        
        # 准备对比数据
        comparison_data = {}
        for strategy, (results, stats) in comparison_results.items():
            comparison_data[strategy] = {
                'NPV均值': stats['npv_mean'],
                'NPV标准差': stats['npv_std'],
                '平均利用率': stats['utilization_mean'],
                '自给自足率': stats['self_sufficiency_mean'],
                '正NPV概率': stats['probability_positive_npv']
            }
        
        # 显示对比表格
        TerminalDisplay.print_comparison_table(comparison_data, f"T={T}年策略对比")
        
        # 显示最佳策略
        best_strategy = max(comparison_data.keys(), 
                          key=lambda s: comparison_data[s]['NPV均值'])
        
        best_data = {
            "最佳策略": best_strategy.title(),
            "NPV均值": comparison_data[best_strategy]['NPV均值'],
            "利用率": comparison_data[best_strategy]['平均利用率'],
            "成功率": comparison_data[best_strategy]['正NPV概率']
        }
        
        TerminalDisplay.print_summary_box(f"T={T}年最佳策略", best_data, 'green')
    
    def _display_horizon_analysis(self, horizon_results: Dict):
        """显示时间跨度分析结果"""
        TerminalDisplay.print_section("时间跨度分析摘要")
        
        # 准备分析数据
        analysis_data = []
        for T, strategies in horizon_results.items():
            for strategy, (results, stats) in strategies.items():
                analysis_data.append({
                    "时间跨度": T,
                    "策略": strategy.title(),
                    "NPV均值": stats['npv_mean'],
                    "NPV标准差": stats['npv_std'],
                    "利用率": stats['utilization_mean'],
                    "成功率": stats['probability_positive_npv']
                })
        
        # 定义表格列
        from strategies.utils.terminal_display import TableColumn
        columns = [
            TableColumn("时间跨度", 8, 'center'),
            TableColumn("策略", 12, 'left'),
            TableColumn("NPV均值", 12, 'right', TerminalDisplay._format_number),
            TableColumn("NPV标准差", 12, 'right', TerminalDisplay._format_number),
            TableColumn("利用率", 10, 'right', lambda x: f"{x:.1%}"),
            TableColumn("成功率", 10, 'right', lambda x: f"{x:.1%}")
        ]
        
        TerminalDisplay.print_table(analysis_data, columns, "时间跨度分析结果")


def load_parameters() -> Dict:
    """加载系统参数"""
    params_path = project_root / "data" / "parameters.json"
    with open(params_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # 测试代码
    print("=== 批量仿真执行器测试 ===")
    
    # 加载参数
    params = load_parameters()
    
    # 创建批量执行器
    runner = BatchSimulationRunner(params)
    
    # 测试单个策略仿真
    print("\n--- 单策略仿真测试 ---")
    results, stats = runner.run_single_batch("upfront_deployment", T=10, n_simulations=20)
    print(f"完成 {len(results)} 次仿真")
    print(f"NPV均值: ${stats['npv_mean']:,.0f}")
    
    # 测试策略对比
    print("\n--- 策略对比测试 ---")
    comparison = runner.run_strategy_comparison(
        strategies=["upfront_deployment", "flexible_deployment"],
        T=10,
        n_simulations=10
    )
    
    # 测试时间跨度分析
    print("\n--- 时间跨度分析测试 ---")
    horizon_analysis = runner.run_time_horizon_analysis(
        time_horizons=[10, 20],
        strategies=["gradual_deployment"],
        n_simulations=5
    )