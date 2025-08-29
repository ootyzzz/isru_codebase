#!/usr/bin/env python3
"""
ISRU策略可视化器
实现多策略决策变量的对比可视化和分析图表生成

主要功能:
    - 决策变量对比图: 生产量、产能、库存、地球供应、产能扩张、利用率
    - 需求与供应对比图: 需求曲线与各策略供应能力对比
    - 成本分析图: 成本构成堆叠图和NPV对比图

使用示例:
    # 基本使用
    plotter = DecisionVariablesPlotter()
    figures = plotter.create_comprehensive_dashboard("strategies/simulation_results", time_horizon=50)
    
    # 单独生成图表
    strategies_data = plotter.load_simulation_data("strategies/simulation_results", 30)
    fig1 = plotter.plot_decision_variables(strategies_data)
    fig2 = plotter.plot_demand_vs_supply(strategies_data)
    fig3 = plotter.plot_cost_analysis(strategies_data)

支持的时间跨度: T10, T20, T30, T40, T50
支持的策略: conservative (保守), moderate (温和), aggressive (激进)
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# 设置matplotlib为交互模式
plt.ion()

# 忽略matplotlib的一些警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class DecisionVariablesPlotter:
    """
    ISRU策略可视化绘图器
    
    用于生成ISRU策略仿真结果的可视化图表，支持多策略对比分析。
    自动适配不同时间跨度的数据，提供中文界面和交互式图表。
    
    Attributes:
        figsize: 图表尺寸 (宽, 高)
        strategy_colors: 策略颜色映射
        strategy_labels: 策略中文标签映射
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """
        初始化绘图器
        
        Args:
            figsize: 图表尺寸 (宽, 高)
        """
        self.figsize = figsize
        self.strategy_colors = {
            'upfront_deployment': '#2E8B57',    # 海绿色 - 一次性部署
            'gradual_deployment': '#4169E1',    # 皇家蓝 - 渐进部署
            'flexible_deployment': '#DC143C'    # 深红色 - 灵活部署
        }
        self.strategy_labels = {
            'upfront_deployment': 'Upfront Deployment',
            'gradual_deployment': 'Gradual Deployment',
            'flexible_deployment': 'Flexible Deployment'
        }
        
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_simulation_data(self, results_dir: str = "strategies/simulation_results", time_horizon: int = 10) -> Dict[str, Any]:
        """
        加载仿真结果数据
        
        Args:
            results_dir: 结果目录路径
            time_horizon: 时间跨度（年）
            
        Returns:
            包含三个策略数据的字典
        """
        results_path = Path(results_dir)
        strategies_data = {}
        
        # 加载三个策略的最新结果
        for strategy in ['upfront_deployment', 'gradual_deployment', 'flexible_deployment']:
            latest_file = results_path / "raw" / f"T{time_horizon}" / f"{strategy}_latest.json"
            
            if latest_file.exists():
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    strategies_data[strategy] = data
            else:
                print(f"Warning: Data file not found for {strategy} strategy: {latest_file}")
                
        return strategies_data
    
    def extract_decision_variables(self, simulation_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        从仿真数据中提取决策变量
        
        Args:
            simulation_data: 单个策略的仿真数据
            
        Returns:
            决策变量字典
        """
        if not simulation_data or 'results' not in simulation_data:
            return {}
            
        # 取第一个仿真结果（通常包含多个随机种子的结果）
        result = simulation_data['results'][0]
        
        # 提取时间序列
        time_steps = list(range(len(result['decisions'])))
        
        # 提取各种决策变量
        variables = {
            'time_steps': time_steps,
            'production': [],           # 生产量
            'capacity': [],            # 产能
            'inventory': [],           # 库存
            'earth_supply': [],        # 地球供应
            'capacity_expansion': [],  # 产能扩张
            'utilization': [],         # 利用率
            'demand': result.get('demand_path', [])  # 需求路径
        }
        
        # 从decisions中提取数据
        for decision in result['decisions']:
            variables['production'].append(decision.get('planned_production', 0))
            variables['capacity_expansion'].append(decision.get('capacity_expansion', 0))
            variables['earth_supply'].append(decision.get('earth_supply_request', 0))
        
        # 从states中提取数据
        for state in result['states']:
            variables['capacity'].append(state.get('total_capacity', 0))
            variables['inventory'].append(state.get('inventory', 0))
            
            # 计算利用率
            capacity = state.get('total_capacity', 1)
            production = state.get('actual_production', 0)
            utilization = production / capacity if capacity > 0 else 0
            variables['utilization'].append(utilization)
        
        return variables
    
    def plot_decision_variables(self, strategies_data: Dict[str, Any], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制决策变量对比图
        
        Args:
            strategies_data: 包含所有策略数据的字典
            save_path: 保存路径（可选）
            
        Returns:
            matplotlib图形对象
        """
        # 创建子图布局 (3行2列)
        fig, axes = plt.subplots(3, 2, figsize=self.figsize)
        fig.suptitle('ISRU Decision Variables Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 扁平化axes数组以便索引
        axes_flat = axes.flatten()
        
        # 定义要绘制的变量和对应的子图
        plot_configs = [
            ('production', 'Production (kg)', 0),
            ('capacity', 'Capacity (kg)', 1),
            ('inventory', 'Inventory Level (kg)', 2),
            ('earth_supply', 'Earth Supply (kg)', 3),
            ('capacity_expansion', 'Capacity Expansion (kg)', 4),
            ('utilization', 'Utilization Rate', 5)
        ]
        
        # 为每个策略提取数据并绘图
        all_variables = {}
        for strategy_name, data in strategies_data.items():
            if data:
                all_variables[strategy_name] = self.extract_decision_variables(data)
        
        # 绘制每个变量
        for var_name, var_label, ax_idx in plot_configs:
            ax = axes_flat[ax_idx]
            
            for strategy_name, variables in all_variables.items():
                if var_name in variables and variables[var_name]:
                    time_steps = variables.get('time_steps', range(len(variables[var_name])))
                    values = variables[var_name]
                    
                    # 确保时间步长和数值长度一致
                    min_len = min(len(time_steps), len(values))
                    time_steps_plot = time_steps[:min_len]
                    values_plot = values[:min_len]
                    
                    ax.plot(time_steps_plot, values_plot, 
                           color=self.strategy_colors[strategy_name],
                           label=self.strategy_labels[strategy_name],
                           linewidth=2, marker='o', markersize=4)
            
            ax.set_title(var_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step (Years)', fontsize=10)
            ax.set_ylabel(var_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # 特殊处理利用率图表
            if var_name == 'utilization':
                ax.set_ylim(0, 1.1)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Full Capacity')
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图表（交互模式）
        plt.show()
        
        # 保存图表（如果指定了路径）
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        return fig
    
    def plot_demand_vs_supply(self, strategies_data: Dict[str, Any]) -> plt.Figure:
        """
        绘制需求与供应对比图 - 修复版本
        
        Args:
            strategies_data: 包含所有策略数据的字典
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        demand_plotted = False  # 标记是否已绘制需求线
        
        for strategy_name, data in strategies_data.items():
            if not data:
                continue
                
            variables = self.extract_decision_variables(data)
            if not variables:
                continue
                
            time_steps = variables.get('time_steps', [])
            demand = variables.get('demand', [])
            production = variables.get('production', [])
            earth_supply = variables.get('earth_supply', [])
            
            # 计算总供应（生产+地球供应）
            total_supply = [p + e for p, e in zip(production, earth_supply)]
            
            # 绘制需求线（所有策略的需求应该相同，只绘制一次）
            if not demand_plotted and demand:
                # 需求数据通常比时间步长多1个（包含初始值），创建对应的时间轴
                demand_time_steps = list(range(len(demand)))
                ax.plot(demand_time_steps, demand, 
                       color='black', linewidth=3, linestyle='--', 
                       label='Demand', alpha=0.8)
                demand_plotted = True
            
            # 绘制总供应线
            min_len = min(len(time_steps), len(total_supply))
            if min_len > 0:
                ax.plot(time_steps[:min_len], total_supply[:min_len],
                       color=self.strategy_colors[strategy_name],
                       label=f'{self.strategy_labels[strategy_name]} - Total Supply',
                       linewidth=2, marker='s', markersize=4)
        
        ax.set_title('Demand vs Supply Comparison Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step (Years)', fontsize=12)
        ax.set_ylabel('Quantity (kg)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_cost_analysis(self, strategies_data: Dict[str, Any]) -> plt.Figure:
        """
        绘制成本分析图
        
        Args:
            strategies_data: 包含所有策略数据的字典
            
        Returns:
            matplotlib图形对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 提取成本数据
        strategies = []
        expansion_costs = []
        operational_costs = []
        supply_costs = []
        total_costs = []
        npvs = []
        
        for strategy_name, data in strategies_data.items():
            if not data or 'results' not in data:
                continue
                
            result = data['results'][0]
            metrics = result.get('performance_metrics', {})
            
            strategies.append(self.strategy_labels[strategy_name])
            expansion_costs.append(metrics.get('total_expansion_cost', 0))
            operational_costs.append(metrics.get('total_operational_cost', 0))
            supply_costs.append(metrics.get('total_supply_cost', 0))
            total_costs.append(metrics.get('total_cost', 0))
            npvs.append(metrics.get('npv', 0))
        
        # 绘制成本构成堆叠柱状图
        x = np.arange(len(strategies))
        width = 0.6
        
        ax1.bar(x, expansion_costs, width, label='Expansion Cost',
               color='#FF6B6B', alpha=0.8)
        ax1.bar(x, operational_costs, width, bottom=expansion_costs,
               label='Operational Cost', color='#4ECDC4', alpha=0.8)
        ax1.bar(x, supply_costs, width,
               bottom=[e+o for e,o in zip(expansion_costs, operational_costs)],
               label='Supply Cost', color='#45B7D1', alpha=0.8)
        
        ax1.set_title('Cost Composition Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost (10K CNY)', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 绘制NPV对比
        colors = [self.strategy_colors[name] for name in strategies_data.keys() if strategies_data[name]]
        bars = ax2.bar(x, npvs, width, color=colors, alpha=0.8)
        
        ax2.set_title('Net Present Value (NPV) Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('NPV (10K CNY)', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar, npv in zip(bars, npvs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{npv/10000:.1f}0K',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_npv_cdf(self, strategies_data: Dict[str, Any]) -> plt.Figure:
        """
        绘制NPV累积分布函数(CDF)图
        
        Args:
            strategies_data: 包含所有策略数据的字典
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for strategy_name, data in strategies_data.items():
            if not data or 'results' not in data:
                continue
                
            # 提取所有仿真的NPV值
            npv_values = []
            for result in data['results']:
                npv = result.get('performance_metrics', {}).get('npv', 0)
                npv_values.append(npv / 10000)  # 转换为万元
            
            if npv_values:
                # 排序NPV值
                npv_sorted = np.sort(npv_values)
                # 计算累积概率
                cumulative_prob = np.arange(1, len(npv_sorted) + 1) / len(npv_sorted)
                
                # 绘制CDF曲线
                ax.plot(npv_sorted, cumulative_prob,
                       color=self.strategy_colors[strategy_name],
                       label=self.strategy_labels[strategy_name],
                       linewidth=2, marker='o', markersize=3, alpha=0.8)
                
                # 添加统计信息
                mean_npv = np.mean(npv_values)
                median_npv = np.median(npv_values)
                
                # 在图上标记均值和中位数
                mean_prob = np.interp(mean_npv, npv_sorted, cumulative_prob)
                median_prob = 0.5
                
                ax.axvline(x=mean_npv, color=self.strategy_colors[strategy_name],
                          linestyle='--', alpha=0.8, linewidth=2,
                          label=f'E[NPV] {self.strategy_labels[strategy_name]}')
                ax.axvline(x=median_npv, color=self.strategy_colors[strategy_name],
                          linestyle=':', alpha=0.6, linewidth=1)
        
        # 添加零NPV参考线
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.5, linewidth=1, label='Break-even (NPV=0)')
        
        ax.set_title('NPV Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
        ax.set_xlabel('NPV (10K CNY)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 设置y轴范围为0-1
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_comprehensive_dashboard(self, results_dir: str = "strategies/simulation_results", time_horizon: int = 10) -> List[plt.Figure]:
        """
        创建综合仪表板
        
        Args:
            results_dir: 结果目录路径
            time_horizon: 时间跨度（年）
            
        Returns:
            图形对象列表
        """
        print("Loading simulation data...")
        strategies_data = self.load_simulation_data(results_dir, time_horizon)
        
        if not strategies_data:
            print("Error: No strategy data found")
            return []
        
        print(f"Loaded data for {len(strategies_data)} strategies")
        
        figures = []
        
        try:
            # 1. 主要决策变量对比图
            print("Generating decision variables comparison chart...")
            fig1 = self.plot_decision_variables(strategies_data)
            figures.append(fig1)
            
            # 2. 需求与供应对比图
            print("Generating demand vs supply comparison chart...")
            fig2 = self.plot_demand_vs_supply(strategies_data)
            figures.append(fig2)
            
            # 3. 成本分析图
            print("Generating cost analysis chart...")
            fig3 = self.plot_cost_analysis(strategies_data)
            figures.append(fig3)
            
            # 4. NPV累积分布函数图
            print("Generating NPV CDF chart...")
            fig4 = self.plot_npv_cdf(strategies_data)
            figures.append(fig4)
            
            print("All charts generated successfully!")
        except Exception as e:
            print(f"Error occurred while generating charts: {e}")
            
        return figures


def main():
    """
    主函数 - 用于测试和独立运行
    
    运行示例:
        python strategies/visualization/strategy_visualizer.py
    """
    print("=== ISRU策略可视化器测试 ===")
    
    # 创建绘图器
    plotter = DecisionVariablesPlotter(figsize=(16, 12))
    
    # 测试不同时间跨度
    for time_horizon in [10, 50]:
        print(f"\n测试时间跨度: T{time_horizon}")
        
        # 生成综合仪表板
        figures = plotter.create_comprehensive_dashboard(
            results_dir="strategies/simulation_results",
            time_horizon=time_horizon
        )
        
        if figures:
            print(f"成功生成 {len(figures)} 个图表")
            
            # 保持图表显示
            input(f"按回车键关闭 T{time_horizon} 的图表...")
            plt.close('all')
        else:
            print(f"未能生成 T{time_horizon} 的图表")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main()