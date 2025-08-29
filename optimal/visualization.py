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

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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

def create_npv_vs_t_visualization(df, output_dir="charts"):
    """
    创建NPV vs T的可视化图表
    
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
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ISRU Project NPV Analysis vs Time Horizon', fontsize=16, fontweight='bold')
    
    # 1. 主图：NPV vs T
    ax1 = axes[0, 0]
    ax1.plot(df_success['T'], df_success['NPV'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Break-even')
    ax1.set_xlabel('Time Horizon (Years)')
    ax1.set_ylabel('NPV (Currency Units)')
    ax1.set_title('NPV vs Time Horizon')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 标注最优点
    max_npv_idx = df_success['NPV'].idxmax()
    max_npv_t = df_success.loc[max_npv_idx, 'T']
    max_npv_value = df_success.loc[max_npv_idx, 'NPV']
    ax1.annotate(f'Max NPV\nT={max_npv_t}, NPV={max_npv_value:,.0f}',
                xy=(max_npv_t, max_npv_value), xytext=(max_npv_t+5, max_npv_value+max_npv_value*0.1),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. NPV增长率
    ax2 = axes[0, 1]
    df_success_sorted = df_success.sort_values('T')
    npv_growth = df_success_sorted['NPV'].pct_change() * 100
    ax2.plot(df_success_sorted['T'].iloc[1:], npv_growth.iloc[1:], 'g-', linewidth=2, marker='s', markersize=4)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Time Horizon (Years)')
    ax2.set_ylabel('NPV Growth Rate (%)')
    ax2.set_title('NPV Period-over-Period Growth Rate')
    ax2.grid(True, alpha=0.3)
    
    # 3. 求解时间分析
    ax3 = axes[1, 0]
    ax3.scatter(df_success['T'], df_success['Solve_Time'], alpha=0.6, s=50)
    ax3.set_xlabel('Time Horizon (Years)')
    ax3.set_ylabel('Solve Time (Seconds)')
    ax3.set_title('Computational Time vs Problem Size')
    ax3.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(df_success['T'], df_success['Solve_Time'], 1)
    p = np.poly1d(z)
    ax3.plot(df_success['T'], p(df_success['T']), "r--", alpha=0.8, linewidth=2)
    
    # 4. NPV分布直方图
    ax4 = axes[1, 1]
    ax4.hist(df_success['NPV'], bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(x=df_success['NPV'].mean(), color='r', linestyle='--', 
                label=f'Mean: {df_success["NPV"].mean():,.0f}')
    ax4.axvline(x=df_success['NPV'].median(), color='g', linestyle='--', 
                label=f'Median: {df_success["NPV"].median():,.0f}')
    ax4.set_xlabel('NPV (Currency Units)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('NPV Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/npv_vs_time_analysis_{timestamp}.png"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"📈 图表已保存到: {output_file}")
    
    # 显示图表
    plt.show()
    
    return output_file

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
        
        # 创建可视化
        chart_file = create_npv_vs_t_visualization(df)
        
        print(f"\n🎯 分析完成！")
        print(f"   数据文件: {data_file}")
        print(f"   图表文件: {chart_file}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        print("\n💡 提示:")
        print("   请先运行批量分析: python optimal/npv_time_analysis.py")

if __name__ == "__main__":
    main()
