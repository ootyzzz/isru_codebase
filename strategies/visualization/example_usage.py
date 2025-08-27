#!/usr/bin/env python3
"""
决策变量可视化使用示例
演示如何使用可视化系统
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.visualization.decision_variables_plotter import DecisionVariablesPlotter


def basic_usage_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 创建可视化器
    plotter = DecisionVariablesPlotter(figsize=(16, 12))
    
    # 加载数据
    print("正在加载仿真数据...")
    strategies_data = plotter.load_simulation_data()
    
    if not strategies_data:
        print("错误: 未找到仿真数据")
        return
    
    print(f"成功加载 {len(strategies_data)} 个策略的数据")
    
    # 生成决策变量对比图
    print("正在生成决策变量对比图...")
    fig1 = plotter.plot_decision_variables(strategies_data)
    
    # 询问是否继续
    response = input("是否继续查看其他图表? (y/n): ").lower().strip()
    if response == 'y':
        # 生成需求与供应对比图
        print("正在生成需求与供应对比图...")
        fig2 = plotter.plot_demand_vs_supply(strategies_data)
        
        # 生成成本分析图
        print("正在生成成本分析图...")
        fig3 = plotter.plot_cost_analysis(strategies_data)
    
    print("示例完成！")


def comprehensive_dashboard_example():
    """综合仪表板示例"""
    print("\n=== 综合仪表板示例 ===")
    
    # 创建可视化器
    plotter = DecisionVariablesPlotter()
    
    # 生成综合仪表板
    figures = plotter.create_comprehensive_dashboard()
    
    if figures:
        print(f"成功生成 {len(figures)} 个图表的综合仪表板")
        
        # 保持图表显示
        input("按回车键关闭所有图表...")
        
        # 关闭所有图表
        import matplotlib.pyplot as plt
        plt.close('all')
    else:
        print("未能生成仪表板")


def custom_visualization_example():
    """自定义可视化示例"""
    print("\n=== 自定义可视化示例 ===")
    
    # 创建自定义尺寸的可视化器
    plotter = DecisionVariablesPlotter(figsize=(20, 15))
    
    # 加载数据
    strategies_data = plotter.load_simulation_data()
    
    if not strategies_data:
        print("错误: 未找到仿真数据")
        return
    
    # 只生成特定的图表
    print("正在生成自定义决策变量图表...")
    
    # 可以保存图表到文件
    save_path = "strategies/visualization/decision_variables_comparison.png"
    fig = plotter.plot_decision_variables(strategies_data, save_path=save_path)
    
    print(f"图表已保存到: {save_path}")


def data_extraction_example():
    """数据提取示例"""
    print("\n=== 数据提取示例 ===")
    
    plotter = DecisionVariablesPlotter()
    strategies_data = plotter.load_simulation_data()
    
    if not strategies_data:
        print("错误: 未找到仿真数据")
        return
    
    # 提取并显示各策略的关键指标
    for strategy_name, data in strategies_data.items():
        if not data:
            continue
            
        print(f"\n{strategy_name.upper()} 策略:")
        
        # 提取决策变量
        variables = plotter.extract_decision_variables(data)
        
        if variables:
            # 显示关键统计信息
            production = variables.get('production', [])
            capacity = variables.get('capacity', [])
            utilization = variables.get('utilization', [])
            
            if production:
                print(f"  平均生产量: {sum(production)/len(production):.2f} kg")
            if capacity:
                print(f"  最终产能: {capacity[-1]:.2f} kg")
            if utilization:
                print(f"  平均利用率: {sum(utilization)/len(utilization):.2%}")
        
        # 显示性能指标
        if 'results' in data and data['results']:
            metrics = data['results'][0].get('performance_metrics', {})
            npv = metrics.get('npv', 0)
            self_sufficiency = metrics.get('self_sufficiency_rate', 0)
            
            print(f"  净现值: {npv/10000:.1f} 万元")
            print(f"  自给自足率: {self_sufficiency:.2%}")


def main():
    """主函数"""
    print("=== ISRU决策变量可视化使用示例 ===")
    
    # 检查数据文件是否存在
    results_dir = Path("strategies/simulation_results")
    if not results_dir.exists():
        print("错误: 仿真结果目录不存在")
        print("请先运行仿真生成数据文件")
        return
    
    print("选择要运行的示例:")
    print("1. 基本使用示例")
    print("2. 综合仪表板示例")
    print("3. 自定义可视化示例")
    print("4. 数据提取示例")
    print("5. 运行所有示例")
    
    choice = input("请输入选择 (1-5): ").strip()
    
    if choice == '1':
        basic_usage_example()
    elif choice == '2':
        comprehensive_dashboard_example()
    elif choice == '3':
        custom_visualization_example()
    elif choice == '4':
        data_extraction_example()
    elif choice == '5':
        basic_usage_example()
        comprehensive_dashboard_example()
        custom_visualization_example()
        data_extraction_example()
    else:
        print("无效选择，运行基本示例...")
        basic_usage_example()
    
    print("\n示例运行完成！")


if __name__ == "__main__":
    main()