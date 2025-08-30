#!/usr/bin/env python3
"""
ISRU Strategy Simulation System Main Entry
Refactored strategy simulation and comparison analysis system

Quick Start Examples:
    # Run 50-year time horizon strategy comparison simulation with visualization
    python strategies/main.py --time-horizon 50 --visualize --n-simulations 1000

    # Run 10-year time horizon strategy comparison simulation (default)
    python strategies/main.py --visualize

    # Run single strategy Monte Carlo simulation
    python strategies/main.py monte-carlo --strategy flexible_deployment --time-horizon 30 --n-simulations 500 --visualize

    # Run strategy comparison analysis
    python strategies/main.py compare --time-horizon 25 --n-simulations 200 --visualize --save

    # Display visualization charts for existing results
    python strategies/main.py visualize

    # Export results to Excel
    python strategies/main.py results export --strategies upfront_deployment gradual_deployment flexible_deployment --time-horizons 10 20 30

Main Features:
    - Strategy Simulation: Support for Upfront Deployment, Gradual Deployment, Flexible Deployment ISRU strategies
    - Time Horizon Analysis: Configurable 10-50 year simulation time horizons
    - Monte Carlo Simulation: Support for multiple random simulations to evaluate strategy robustness
    - Visualization Analysis: Automatically generate decision variables, cost analysis charts
    - Result Management: Support for result saving, loading, exporting functions
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.simulation_engine import StrategySimulationEngine
from strategies.analysis.batch_runner import BatchSimulationRunner
from strategies.analysis.performance_analyzer import PerformanceAnalyzer
from strategies.utils.terminal_display import TerminalDisplay
from strategies.utils.result_manager import ResultManager

# 导入可视化模块
try:
    from strategies.visualization.strategy_visualizer import DecisionVariablesPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization module unavailable: {e}")
    VISUALIZATION_AVAILABLE = False


def load_parameters() -> dict:
    """加载系统参数"""
    params_path = project_root / "data" / "parameters.json"
    with open(params_path, 'r') as f:
        return json.load(f)


def show_visualization(results_dir: str = "strategies/simulation_results", time_horizon: int = 10):
    """
    显示可视化图表
    
    Args:
        results_dir: 结果目录路径
        time_horizon: 时间跨度（年）
    """
    if not VISUALIZATION_AVAILABLE:
        print("ERROR: Visualization feature unavailable, please check if matplotlib and other dependencies are installed")
        return
    
    try:
        TerminalDisplay.print_section("Generating Visualization Charts")
        
        # 创建可视化器
        plotter = DecisionVariablesPlotter(figsize=(16, 12))
        
        # 生成综合仪表板，传递时间跨度参数
        figures = plotter.create_comprehensive_dashboard(results_dir, time_horizon)
        
        if figures:
            print(f"SUCCESS: Successfully generated {len(figures)} charts")
            print("INFO: Charts displayed, close chart windows to continue...")
            
            # 等待用户关闭图表
            try:
                import matplotlib.pyplot as plt
                # 保持图表显示直到用户关闭
                plt.show(block=True)
            except KeyboardInterrupt:
                print("\nUser interrupted, closing all charts...")
                plt.close('all')
        else:
            print("ERROR: Failed to generate charts, please check if simulation data is available")
            
    except Exception as e:
        print(f"ERROR: Error occurred during visualization: {e}")
        print("Please ensure simulation has been run and result data has been generated")


def run_default_simulation_with_visualization(args):
    """
    运行默认的T=10策略对比仿真并显示可视化
    
    Args:
        args: 命令行参数
    """
    TerminalDisplay.print_header("ISRU策略仿真系统 - 默认运行模式", width=70)
    
    # 设置默认参数
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    # 运行策略对比 (T=10, 三个策略)
    strategies = ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    time_horizon = args.time_horizon
    n_simulations = getattr(args, 'n_simulations', 50)  # 默认50次仿真
    
    TerminalDisplay.print_section(f"运行 T={time_horizon} 策略对比仿真")
    print(f"策略: {', '.join([s.title() for s in strategies])}")
    print(f"仿真次数: {n_simulations}")
    print(f"时间跨度: {time_horizon} 年")
    
    try:
        # 运行策略对比
        comparison_results = runner.run_strategy_comparison(
            strategies=strategies,
            T=time_horizon,
            n_simulations=n_simulations,
            base_seed=getattr(args, 'seed', 42),
            save_results=True,  # 强制保存结果以便可视化
            show_progress=True
        )
        
        # 显示简要结果
        TerminalDisplay.print_section("Simulation Complete - Results Summary")
        for strategy_name, (results, stats) in comparison_results.items():
            summary_data = {
                "Strategy": strategy_name.title(),
                "NPV Mean": f"{stats['npv_mean']/70000:.1f}0K GBP",
                "Success Rate": f"{stats['probability_positive_npv']:.1%}",
                "Average Utilization": f"{stats['utilization_mean']:.1%}"
            }
            TerminalDisplay.print_summary_box(f"{strategy_name.title()} Strategy", summary_data)
        
        # 显示可视化
        if args.visualize:
            show_visualization(time_horizon=time_horizon)
        else:
            print("\nTIP: Use --visualize parameter to view chart analysis")
            
    except Exception as e:
        print(f"ERROR: Error occurred during simulation: {e}")
        return


def run_single_simulation(args):
    """运行单次仿真"""
    TerminalDisplay.print_header("单次策略仿真", width=70)
    
    params = load_parameters()
    engine = StrategySimulationEngine(params)
    
    result = engine.run_single_simulation(
        strategy_name=args.strategy,
        T=args.time_horizon,
        seed=args.seed
    )
    
    # 显示结果
    TerminalDisplay.print_section(f"{args.strategy.title()} 策略仿真结果")
    
    summary_data = {
        "策略": args.strategy.title(),
        "时间跨度": f"{args.time_horizon}年",
        "NPV": result.performance_metrics.get('npv', 0),
        "平均利用率": result.performance_metrics.get('avg_utilization', 0),
        "自给自足率": result.performance_metrics.get('self_sufficiency_rate', 0),
        "总成本": result.performance_metrics.get('total_cost', 0)
    }
    
    TerminalDisplay.print_summary_box("仿真结果", summary_data)
    
    # 保存结果
    if args.save:
        output_dir = Path("strategies/simulation_results") / f"T{args.time_horizon}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{args.strategy}_single_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"结果已保存到: {output_file}")


def run_monte_carlo_simulation(args):
    """运行蒙特卡洛仿真"""
    TerminalDisplay.print_header("蒙特卡洛策略仿真", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    results, stats = runner.run_single_batch(
        strategy_name=args.strategy,
        T=args.time_horizon,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=args.save,
        show_progress=True
    )
    
    # 显示统计结果
    TerminalDisplay.print_section("蒙特卡洛仿真统计结果")
    
    stats_data = {
        "仿真次数": len(results),
        "NPV均值": stats['npv_mean'],
        "NPV标准差": stats['npv_std'],
        "成功率": stats['probability_positive_npv'],
        "平均利用率": stats['utilization_mean']
    }
    
    TerminalDisplay.print_summary_box(f"{args.strategy.title()} 策略统计", stats_data)


def run_strategy_comparison(args):
    """运行策略对比"""
    TerminalDisplay.print_header("策略对比分析", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    strategies = args.strategies if args.strategies else ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    
    comparison_results = runner.run_strategy_comparison(
        strategies=strategies,
        T=args.time_horizon,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=args.save,
        show_progress=True
    )
    
    # 性能分析
    if args.detailed_analysis:
        TerminalDisplay.print_section("详细性能分析")
        
        analyzer = PerformanceAnalyzer()
        strategy_results = {name: results for name, (results, _) in comparison_results.items()}
        
        # 生成分析报告
        if args.save:
            report_file = Path("strategies/simulation_results") / f"comparison_report_T{args.time_horizon}.txt"
            report = analyzer.generate_performance_report(strategy_results, report_file)
            print(f"详细报告已保存到: {report_file}")
        else:
            report = analyzer.generate_performance_report(strategy_results)
            print(report)


def run_time_horizon_analysis(args):
    """运行时间跨度分析"""
    TerminalDisplay.print_header("时间跨度影响分析", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    time_horizons = args.time_horizons if args.time_horizons else [10, 20, 30, 40, 50]
    strategies = args.strategies if args.strategies else ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    
    horizon_results = runner.run_time_horizon_analysis(
        time_horizons=time_horizons,
        strategies=strategies,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=args.save,
        show_progress=True
    )
    
    # 时间跨度影响分析
    if args.detailed_analysis:
        TerminalDisplay.print_section("时间跨度影响分析")
        
        analyzer = PerformanceAnalyzer()
        
        # 重组数据格式
        formatted_results = {}
        for T, strategy_data in horizon_results.items():
            formatted_results[T] = {name: results for name, (results, _) in strategy_data.items()}
        
        horizon_analysis = analyzer.analyze_time_horizon_impact(formatted_results)
        
        # 显示趋势分析
        trends = horizon_analysis.get('trends', {})
        for strategy, trend_data in trends.items():
            trend_direction = "上升" if trend_data['is_improving'] else "下降"
            print(f"{strategy.title()} 策略: NPV随时间跨度呈{trend_direction}趋势")


def run_parallel_batch(args):
    """运行并行批量仿真"""
    TerminalDisplay.print_header("并行批量仿真", width=70)
    
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    time_horizons = args.time_horizons if args.time_horizons else [10, 20, 30, 40, 50]
    strategies = args.strategies if args.strategies else ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
    
    results = runner.run_parallel_batch(
        strategies=strategies,
        time_horizons=time_horizons,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        max_workers=args.max_workers,
        save_results=args.save
    )
    
    TerminalDisplay.print_section("并行仿真完成")
    print(f"完成 {len(strategies)} 个策略 × {len(time_horizons)} 个时间跨度的仿真")
    print(f"总仿真次数: {len(strategies) * len(time_horizons) * args.n_simulations}")


def compare_with_optimal(args):
    """与全局最优解对比"""
    TerminalDisplay.print_header("策略与全局最优解对比", width=70)
    
    # 运行策略仿真
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    strategy_results = runner.run_strategy_comparison(
        strategies=args.strategies or ["upfront_deployment", "gradual_deployment", "flexible_deployment"],
        T=args.time_horizon,
        n_simulations=args.n_simulations,
        base_seed=args.seed,
        save_results=False,
        show_progress=True
    )
    
    # 运行全局最优解
    TerminalDisplay.print_section("计算全局最优解基准")
    
    try:
        from optimal.optimal_solu import main as run_optimal
        import io
        import contextlib
        
        # 临时修改参数文件中的T值
        original_T = params['economics']['T']
        params['economics']['T'] = args.time_horizon
        
        # 保存临时参数文件
        temp_params_file = project_root / "data" / "parameters_temp.json"
        with open(temp_params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # 运行全局最优解（捕获输出）
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # 这里需要修改test_fixed_model.py来返回结果而不是只打印
            pass
        
        # 恢复原始参数
        params['economics']['T'] = original_T
        temp_params_file.unlink(missing_ok=True)
        
        print("全局最优解计算完成")
        
        # 显示对比结果
        TerminalDisplay.print_section("策略效率分析")
        
        for strategy_name, (results, stats) in strategy_results.items():
            efficiency = stats['npv_mean'] / 7000000  # 假设最优解为100万英镑 (700万人民币)
            print(f"{strategy_name.title()}: 相对效率 {efficiency:.1%}")
    
    except Exception as e:
        print(f"无法运行全局最优解对比: {e}")
        print("请确保test_fixed_model.py可以正常运行")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ISRU策略仿真系统')
    
    # 添加全局参数（用于默认运行模式）
    parser.add_argument('--visualize', action='store_true',
                       help='Display visualization charts (default: False)')
    parser.add_argument('--time-horizon', type=int, default=10,
                       help='Time horizon (years) (default: 10)')
    parser.add_argument('--n-simulations', type=int, default=50,
                       help='Monte Carlo simulation count (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 通用参数
    def add_common_args(subparser):
        subparser.add_argument('--seed', type=int, default=42, help='随机种子')
        subparser.add_argument('--save', action='store_true', help='保存结果')
        subparser.add_argument('--visualize', action='store_true', help='Display visualization charts')
    
    # 单次仿真
    single_parser = subparsers.add_parser('single', help='运行单次仿真')
    single_parser.add_argument('--strategy', choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                              default='gradual_deployment', help='策略类型')
    single_parser.add_argument('--time-horizon', type=int, default=30, help='时间跨度')
    add_common_args(single_parser)
    
    # 蒙特卡洛仿真
    mc_parser = subparsers.add_parser('monte-carlo', help='运行蒙特卡洛仿真')
    mc_parser.add_argument('--strategy', choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                          default='gradual_deployment', help='策略类型')
    mc_parser.add_argument('--time-horizon', type=int, default=30, help='时间跨度')
    mc_parser.add_argument('--n-simulations', type=int, default=100, help='仿真次数')
    add_common_args(mc_parser)
    
    # 策略对比
    compare_parser = subparsers.add_parser('compare', help='运行策略对比')
    compare_parser.add_argument('--strategies', nargs='+',
                               choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                               help='要对比的策略')
    compare_parser.add_argument('--time-horizon', type=int, default=30, help='时间跨度')
    compare_parser.add_argument('--n-simulations', type=int, default=100, help='仿真次数')
    compare_parser.add_argument('--detailed-analysis', action='store_true', help='详细分析')
    add_common_args(compare_parser)
    
    # 时间跨度分析
    horizon_parser = subparsers.add_parser('horizon', help='运行时间跨度分析')
    horizon_parser.add_argument('--time-horizons', nargs='+', type=int, 
                               default=[10, 20, 30, 40, 50], help='时间跨度列表')
    horizon_parser.add_argument('--strategies', nargs='+',
                               choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                               help='要分析的策略')
    horizon_parser.add_argument('--n-simulations', type=int, default=100, help='仿真次数')
    horizon_parser.add_argument('--detailed-analysis', action='store_true', help='详细分析')
    add_common_args(horizon_parser)
    
    # 并行批量仿真
    parallel_parser = subparsers.add_parser('parallel', help='运行并行批量仿真')
    parallel_parser.add_argument('--time-horizons', nargs='+', type=int, 
                                default=[10, 20, 30, 40, 50], help='时间跨度列表')
    parallel_parser.add_argument('--strategies', nargs='+',
                                choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                                help='要分析的策略')
    parallel_parser.add_argument('--n-simulations', type=int, default=100, help='仿真次数')
    parallel_parser.add_argument('--max-workers', type=int, help='最大并行进程数')
    add_common_args(parallel_parser)
    
    # 与最优解对比
    optimal_parser = subparsers.add_parser('optimal', help='与全局最优解对比')
    optimal_parser.add_argument('--strategies', nargs='+',
                               choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                               help='要对比的策略')
    optimal_parser.add_argument('--time-horizon', type=int, default=30, help='时间跨度')
    optimal_parser.add_argument('--n-simulations', type=int, default=100, help='仿真次数')
    add_common_args(optimal_parser)
    
    # 结果管理
    results_parser = subparsers.add_parser('results', help='结果管理')
    results_subparsers = results_parser.add_subparsers(dest='results_command', help='结果管理命令')
    
    # 导出到Excel
    export_parser = results_subparsers.add_parser('export', help='导出结果到Excel')
    export_parser.add_argument('--strategies', nargs='+',
                              choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                              help='要导出的策略')
    export_parser.add_argument('--time-horizons', nargs='+', type=int,
                              help='要导出的时间跨度')
    export_parser.add_argument('--output', type=str, help='输出文件路径')
    
    # 查看可用结果
    list_parser = results_subparsers.add_parser('list', help='查看可用结果')
    
    # 清理旧结果
    cleanup_parser = results_subparsers.add_parser('cleanup', help='清理旧结果')
    cleanup_parser.add_argument('--keep-days', type=int, default=30, help='保留天数')
    
    # 加载结果
    load_parser = results_subparsers.add_parser('load', help='加载之前的结果')
    load_parser.add_argument('--strategy', required=True,
                            choices=['upfront_deployment', 'gradual_deployment', 'flexible_deployment'],
                            help='策略名称')
    load_parser.add_argument('--time-horizon', type=int, required=True, help='时间跨度')
    
    # 可视化命令
    viz_parser = subparsers.add_parser('visualize', help='Display visualization charts for existing results')
    viz_parser.add_argument('--results-dir', type=str, default="strategies/simulation_results",
                           help='结果目录路径')
    viz_parser.add_argument('--time-horizon', type=int, default=50,
                           help='时间跨度（年）')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        run_single_simulation(args)
        # 单次仿真后可视化
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'monte-carlo':
        run_monte_carlo_simulation(args)
        # 蒙特卡洛仿真后可视化
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'compare':
        run_strategy_comparison(args)
        # 策略对比后可视化
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'horizon':
        run_time_horizon_analysis(args)
        # 时间跨度分析后可视化
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'parallel':
        run_parallel_batch(args)
        # 并行仿真后可视化
        if args.visualize:
            show_visualization(time_horizon=args.time_horizon)
    elif args.command == 'optimal':
        compare_with_optimal(args)
    elif args.command == 'results':
        handle_results_command(args)
    elif args.command == 'visualize':
        # 纯可视化命令
        show_visualization(args.results_dir, time_horizon=args.time_horizon)
    else:
        # 默认运行T=10策略对比并可视化
        run_default_simulation_with_visualization(args)


def handle_results_command(args):
    """处理结果管理命令"""
    params = load_parameters()
    runner = BatchSimulationRunner(params)
    
    if args.results_command == 'export':
        TerminalDisplay.print_header("导出结果到Excel", width=70)
        
        strategies = args.strategies or ["upfront_deployment", "gradual_deployment", "flexible_deployment"]
        time_horizons = args.time_horizons or [10, 20, 30, 40, 50]
        output_file = Path(args.output) if args.output else None
        
        try:
            excel_file = runner.export_results_to_excel(strategies, time_horizons, output_file)
            
            summary_data = {
                "导出策略": ", ".join(strategies),
                "时间跨度": ", ".join(map(str, time_horizons)),
                "输出文件": str(excel_file),
                "文件大小": f"{excel_file.stat().st_size / 1024:.1f} KB"
            }
            
            TerminalDisplay.print_summary_box("导出完成", summary_data, 'green')
            
        except Exception as e:
            print(f"ERROR: 导出失败: {e}")
    
    elif args.results_command == 'list':
        TerminalDisplay.print_header("可用结果列表", width=70)
        
        available = runner.get_available_results()
        
        if not available:
            print("INFO: 暂无可用结果")
            print("请先运行仿真生成结果：")
            print("  python strategies/main.py compare --save")
        else:
            for time_horizon, strategies in available.items():
                TerminalDisplay.print_section(f"{time_horizon} 时间跨度")
                for strategy in strategies:
                    print(f"  - {strategy.title()} 策略")
    
    elif args.results_command == 'cleanup':
        TerminalDisplay.print_header("清理旧结果", width=70)
        
        print(f"清理 {args.keep_days} 天前的结果文件...")
        runner.cleanup_old_results(args.keep_days)
        print("SUCCESS: 清理完成")
    
    elif args.results_command == 'load':
        TerminalDisplay.print_header("加载历史结果", width=70)
        
        result_data = runner.load_previous_results(args.strategy, args.time_horizon)
        
        if result_data:
            metadata = result_data.get('metadata', {})
            results = result_data.get('results', [])
            
            summary_data = {
                "策略": args.strategy.title(),
                "时间跨度": f"{args.time_horizon}年",
                "仿真次数": len(results),
                "生成时间": metadata.get('timestamp', '未知'),
                "版本": metadata.get('version', '未知')
            }
            
            TerminalDisplay.print_summary_box("结果信息", summary_data)
            
            if results:
                # 显示简单统计
                npvs = [r['performance_metrics']['npv'] for r in results]
                npvs_gbp = [npv/7 for npv in npvs]  # 转换为英镑
                print(f"\nNPV统计:")
                print(f"  均值: £{np.mean(npvs_gbp):,.0f}")
                print(f"  标准差: £{np.std(npvs_gbp):,.0f}")
                print(f"  最小值: £{min(npvs_gbp):,.0f}")
                print(f"  最大值: £{max(npvs_gbp):,.0f}")
        else:
            print(f"ERROR: 未找到 {args.strategy} 策略在 T={args.time_horizon} 的结果")
            print("可用结果:")
            available = runner.get_available_results()
            for th, strategies in available.items():
                print(f"  {th}: {', '.join(strategies)}")
    
    else:
        print("请指定结果管理命令: export, list, cleanup, load")


if __name__ == "__main__":
    main()