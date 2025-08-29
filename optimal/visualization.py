"""
NPV vs T 可视化脚本
读取批量分析结果，生成可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime

# 设置matplotlib后端和图表样式
import matplotlib
matplotlib.use('TkAgg')  # 确保使用正确的后端
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100  # 降低DPI以提高性能
plt.rcParams['savefig.dpi'] = 300
# 不设置 plt.ioff()，保持默认的交互模式

def load_latest_results(results_dir="optimal/results"):
    """
    加载最新的分析结果文件
    
    Args:
        results_dir: 结果目录
        
    Returns:
        pandas.DataFrame: 分析结果数据
        str: 文件路径
    """
    # 查找所有结果文件
    pattern = f"{results_dir}/npv_vs_time_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"在目录 {results_dir} 中未找到结果文件")
    
    # 选择最新的文件
    latest_file = max(files, key=os.path.getctime)
    print(f"📂 加载数据文件: {latest_file}")
    
    # 读取数据
    df = pd.read_csv(latest_file)
    print(f"📊 数据概览: {len(df)} 个案例")
    
    return df, latest_file

def create_simple_npv_plot(df, output_dir="optimal/charts"):
    """
    创建简化的NPV vs T主图
    
    Args:
        df: 包含分析结果的DataFrame
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤成功求解的案例
    df_success = df[df['Status'] == 'optimal'].copy()
    
    if df_success.empty:
        print("❌ 没有成功求解的案例，无法生成图表")
        return
    
    # 创建单一主图
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # NPV vs T 主图
    ax.plot(df_success['T'], df_success['NPV'], 'b-', linewidth=3, marker='o', markersize=6, label='NPV')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Break-even')
    ax.set_xlabel('Time Horizon (Years)', fontsize=12)
    ax.set_ylabel('NPV (Currency Units)', fontsize=12)
    ax.set_title('ISRU Project NPV vs Time Horizon', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # 标注最优点
    max_npv_idx = df_success['NPV'].idxmax()
    max_npv_t = df_success.loc[max_npv_idx, 'T']
    max_npv_value = df_success.loc[max_npv_idx, 'NPV']
    ax.annotate(f'Optimal Point\nT={max_npv_t} years\nNPV={max_npv_value:,.0f}',
                xy=(max_npv_t, max_npv_value),
                xytext=(max_npv_t+5, max_npv_value+max_npv_value*0.15),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/npv_main_plot_{timestamp}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"📈 主图已保存到: {output_file}")
    
    # 显示图表
    plt.show()
    
    return output_file

def create_npv_vs_t_visualization(df, output_dir="optimal/charts"):
    """
    创建NPV vs T的可视化图表 - 分别显示在4个独立窗口
    
    Args:
        df: 包含分析结果的DataFrame
        output_dir: 输出目录
    """
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤成功求解的案例
    df_success = df[df['Status'] == 'optimal'].copy()
    
    if df_success.empty:
        print("❌ 没有成功求解的案例，无法生成图表")
        return
    
    print(f"📈 成功案例数量: {len(df_success)}")
    
    # 关闭所有现有图表
    plt.close('all')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files = []
    
    # 使用matplotlib默认行为
    
    # 1. 主图：NPV vs T
    fig1 = plt.figure(1, figsize=(10, 6))
    plt.plot(df_success['T'], df_success['NPV'], 'b-', linewidth=2, marker='o', markersize=6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break-even')
    plt.xlabel('Time Horizon (Years)', fontsize=12)
    plt.ylabel('NPV (Currency Units)', fontsize=12)
    plt.title('NPV vs Time Horizon', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 标注最优点
    max_npv_idx = df_success['NPV'].idxmax()
    max_npv_t = df_success.loc[max_npv_idx, 'T']
    max_npv_value = df_success.loc[max_npv_idx, 'NPV']
    plt.annotate(f'Max NPV\nT={max_npv_t}, NPV={max_npv_value:,.0f}',
                xy=(max_npv_t, max_npv_value), xytext=(max_npv_t+5, max_npv_value+max_npv_value*0.1),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    output_file1 = f"{output_dir}/npv_main_plot_{timestamp}.png"
    plt.savefig(output_file1, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file1)
    print(f"📈 主图已保存到: {output_file1}")
    
    # 2. NPV增长率
    fig2 = plt.figure(2, figsize=(10, 6))
    df_success_sorted = df_success.sort_values('T')
    npv_growth = df_success_sorted['NPV'].pct_change() * 100
    plt.plot(df_success_sorted['T'].iloc[1:], npv_growth.iloc[1:], 'g-', linewidth=2, marker='s', markersize=6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Time Horizon (Years)', fontsize=12)
    plt.ylabel('NPV Growth Rate (%)', fontsize=12)
    plt.title('NPV Period-over-Period Growth Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file2 = f"{output_dir}/npv_growth_rate_{timestamp}.png"
    plt.savefig(output_file2, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file2)
    print(f"📈 增长率图已保存到: {output_file2}")
    
    # 3. 求解时间分析
    fig3 = plt.figure(3, figsize=(10, 6))
    plt.scatter(df_success['T'], df_success['Solve_Time'], alpha=0.6, s=60, color='blue')
    plt.xlabel('Time Horizon (Years)', fontsize=12)
    plt.ylabel('Solve Time (Seconds)', fontsize=12)
    plt.title('Computational Time vs Problem Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(df_success['T'], df_success['Solve_Time'], 1)
    p = np.poly1d(z)
    plt.plot(df_success['T'], p(df_success['T']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    plt.legend()
    
    plt.tight_layout()
    output_file3 = f"{output_dir}/solve_time_analysis_{timestamp}.png"
    plt.savefig(output_file3, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file3)
    print(f"📈 求解时间图已保存到: {output_file3}")
    
    # 4. NPV分布直方图
    fig4 = plt.figure(4, figsize=(10, 6))
    plt.hist(df_success['NPV'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(x=df_success['NPV'].mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {df_success["NPV"].mean():,.0f}')
    plt.axvline(x=df_success['NPV'].median(), color='g', linestyle='--', linewidth=2,
                label=f'Median: {df_success["NPV"].median():,.0f}')
    plt.xlabel('NPV (Currency Units)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('NPV Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file4 = f"{output_dir}/npv_distribution_{timestamp}.png"
    plt.savefig(output_file4, bbox_inches='tight', dpi=300, facecolor='white')
    output_files.append(output_file4)
    print(f"📈 分布图已保存到: {output_file4}")
    
    # 显示所有图表 - 使用默认行为
    plt.show()
    
    print(f"\n🎯 已生成4个独立的图表窗口!")
    
    return output_files

def print_analysis_summary(df):
    """
    打印分析摘要
    
    Args:
        df: 分析结果DataFrame
    """
    print("\n" + "="*60)
    print("📊 NPV vs T 分析摘要")
    print("="*60)
    
    # 基本统计
    df_success = df[df['Status'] == 'optimal']
    total_cases = len(df)
    success_cases = len(df_success)
    
    print(f"总案例数: {total_cases}")
    print(f"成功求解: {success_cases} ({success_cases/total_cases*100:.1f}%)")
    
    if success_cases > 0:
        print(f"\nNPV统计:")
        print(f"  最小值: {df_success['NPV'].min():,.2f}")
        print(f"  最大值: {df_success['NPV'].max():,.2f}")
        print(f"  平均值: {df_success['NPV'].mean():,.2f}")
        print(f"  中位数: {df_success['NPV'].median():,.2f}")
        
        # 找出最优T值
        max_npv_idx = df_success['NPV'].idxmax()
        optimal_t = df_success.loc[max_npv_idx, 'T']
        optimal_npv = df_success.loc[max_npv_idx, 'NPV']
        print(f"\n最优时间长度:")
        print(f"  T = {optimal_t} 年")
        print(f"  NPV = {optimal_npv:,.2f}")
        
        # NPV为正的案例
        positive_npv = df_success[df_success['NPV'] > 0]
        if not positive_npv.empty:
            min_positive_t = positive_npv['T'].min()
            print(f"\nNPV转正点: T = {min_positive_t} 年")
        
        # 求解时间统计
        print(f"\n求解时间统计:")
        print(f"  平均时间: {df_success['Solve_Time'].mean():.3f} 秒")
        print(f"  最长时间: {df_success['Solve_Time'].max():.3f} 秒")
    
    # 失败案例
    failed_cases = df[df['Status'] != 'optimal']
    if not failed_cases.empty:
        print(f"\n失败案例: {len(failed_cases)} 个")
        print(f"失败的T值: {failed_cases['T'].tolist()}")

def main():
    """主函数"""
    print("NPV vs T 可视化分析工具")
    print("=" * 60)
    
    try:
        # 加载数据
        df, data_file = load_latest_results()
        
        # 打印分析摘要
        print_analysis_summary(df)
        
        # 直接创建4个独立窗口的可视化
        print("\n📊 正在生成4个独立图表窗口...")
        chart_files = create_npv_vs_t_visualization(df)
        chart_file = f"4个图表文件: {', '.join([f.split('/')[-1] for f in chart_files])}"
        
        print(f"\n🎯 分析完成！")
        print(f"   数据文件: {data_file}")
        print(f"   图表文件: {chart_file}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        print("\n💡 提示:")
        print("   请先运行批量分析: python optimal/npv_time_analysis.py")

if __name__ == "__main__":
    main()
