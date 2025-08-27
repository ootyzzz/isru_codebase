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
            'conservative': '#2E8B57',  # 海绿色 - 保守
            'moderate': '#4169E1',      # 皇家蓝 - 温和  
            'aggressive': '#DC143C'     # 深红色 - 激进
        }
        self.strategy_labels = {
            'conservative': '保守策略',
            'moderate': '温和策略',
            'aggressive': '激进策略'
        }
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
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
        for strategy in ['conservative', 'moderate', 'aggressive']:
            latest_file = results_path / "raw" / f"T{time_horizon}" / f"{strategy}_latest.json"
            
            if latest_file.exists():
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    strategies_data[strategy] = data
            else:
                print(f"警告: 未找到 {strategy} 策略的数据文件: {latest_file}")
                
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
        fig.suptitle('ISRU决策变量对比分析', fontsize=16, fontweight='bold')
        
        # 扁平化axes数组以便索引
        axes_flat = axes.flatten()
        
        # 定义要绘制的变量和对应的子图
        plot_configs = [
            ('production', '生产量 (kg)', 0),
            ('capacity', '产能 (kg)', 1), 
            ('inventory', '库存水平 (kg)', 2),
            ('earth_supply', '地球供应 (kg)', 3),
            ('capacity_expansion', '产能扩张 (kg)', 4),
            ('utilization', '利用率', 5)
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
            ax.set_xlabel('时间步长 (年)', fontsize=10)
            ax.set_ylabel(var_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # 特殊处理利用率图表
            if var_name == 'utilization':
                ax.set_ylim(0, 1.1)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='满负荷')
        
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
                       label='需求', alpha=0.8)
                demand_plotted = True
            
            # 绘制总供应线
            min_len = min(len(time_steps), len(total_supply))
            if min_len > 0:
                ax.plot(time_steps[:min_len], total_supply[:min_len],
                       color=self.strategy_colors[strategy_name],
                       label=f'{self.strategy_labels[strategy_name]} - 总供应',
                       linewidth=2, marker='s', markersize=4)
        
        ax.set_title('需求与供应对比分析', fontsize=14, fontweight='bold')
        ax.set_xlabel('时间步长 (年)', fontsize=12)
        ax.set_ylabel('数量 (kg)', fontsize=12)
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
        
        ax1.bar(x, expansion_costs, width, label='扩张成本', 
               color='#FF6B6B', alpha=0.8)
        ax1.bar(x, operational_costs, width, bottom=expansion_costs, 
               label='运营成本', color='#4ECDC4', alpha=0.8)
        ax1.bar(x, supply_costs, width, 
               bottom=[e+o for e,o in zip(expansion_costs, operational_costs)],
               label='供应成本', color='#45B7D1', alpha=0.8)
        
        ax1.set_title('成本构成对比', fontsize=12, fontweight='bold')
        ax1.set_ylabel('成本 (万元)', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 绘制NPV对比
        colors = [self.strategy_colors[name] for name in strategies_data.keys() if strategies_data[name]]
        bars = ax2.bar(x, npvs, width, color=colors, alpha=0.8)
        
        ax2.set_title('净现值(NPV)对比', fontsize=12, fontweight='bold')
        ax2.set_ylabel('NPV (万元)', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar, npv in zip(bars, npvs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{npv/10000:.1f}万',
                    ha='center', va='bottom', fontsize=9)
        
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
        print("正在加载仿真数据...")
        strategies_data = self.load_simulation_data(results_dir, time_horizon)
        
        if not strategies_data:
            print("错误: 未找到任何策略数据")
            return []
        
        print(f"已加载 {len(strategies_data)} 个策略的数据")
        
        figures = []
        
        try:
            # 1. 主要决策变量对比图
            print("正在生成决策变量对比图...")
            fig1 = self.plot_decision_variables(strategies_data)
            figures.append(fig1)
            
            # 2. 需求与供应对比图
            print("正在生成需求与供应对比图...")
            fig2 = self.plot_demand_vs_supply(strategies_data)
            figures.append(fig2)
            
            # 3. 成本分析图
            print("正在生成成本分析图...")
            fig3 = self.plot_cost_analysis(strategies_data)
            figures.append(fig3)
            
            print("所有图表已生成完成！")
        except Exception as e:
            print(f"生成图表时出现错误: {e}")
            
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