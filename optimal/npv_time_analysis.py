"""
NPV vs T 批量分析脚本
对T=1到50年进行批量优化求解，收集NPV数据用于后续分析
"""

import json
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入求解函数
from optimal.optimal_solu import solve_isru_optimization

def batch_solve_npv_analysis(t_min=1, t_max=50, random_seed=42, output_dir="optimal/results"):
    """
    批量求解不同T值的ISRU优化问题
    
    Args:
        t_min: 最小T值
        t_max: 最大T值  
        random_seed: 随机种子
        output_dir: 输出目录
        
    Returns:
        pandas.DataFrame: 包含所有结果的数据框
    """
    
    print(f"🚀 开始批量NPV分析：T={t_min}到{t_max}年")
    print(f"📁 结果将保存到: {output_dir}")
    print("="*60)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储结果的列表
    results = []
    failed_cases = []
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 批量求解
    for T in range(t_min, t_max + 1):
        print(f"📊 正在求解 T={T}年...")
        
        try:
            # 求解单个案例
            start_time = time.time()
            result = solve_isru_optimization(T, random_seed=random_seed, verbose=False)
            solve_time = time.time() - start_time
            
            # 记录结果
            result_record = {
                'T': T,
                'NPV': result['npv'],
                'Status': result['status'],
                'Solve_Time': solve_time,
                'Demand_Sum': sum(result['demand_path']) if result['demand_path'] is not None else None,
                'Demand_Mean': np.mean(result['demand_path']) if result['demand_path'] is not None else None,
                'Demand_Std': np.std(result['demand_path']) if result['demand_path'] is not None else None
            }
            
            results.append(result_record)
            
            # 打印进度
            if result['status'] == 'optimal':
                print(f"   ✅ NPV = {result['npv']:,.2f} (用时: {solve_time:.2f}s)")
            else:
                print(f"   ❌ 失败: {result['status']} (用时: {solve_time:.2f}s)")
                failed_cases.append(T)
                
        except Exception as e:
            print(f"   💥 异常: {str(e)}")
            failed_cases.append(T)
            results.append({
                'T': T,
                'NPV': None,
                'Status': 'error',
                'Solve_Time': None,
                'Demand_Sum': None,
                'Demand_Mean': None,
                'Demand_Std': None
            })
    
    # 计算总用时
    total_time = time.time() - total_start_time
    
    # 转换为DataFrame
    df_results = pd.DataFrame(results)
    
    # 打印汇总统计
    print("\n" + "="*60)
    print("📈 批量分析完成！")
    print(f"⏱️  总用时: {total_time:.2f}秒")
    print(f"📊 成功求解: {len(df_results[df_results['Status'] == 'optimal'])} / {len(df_results)}")
    
    if failed_cases:
        print(f"❌ 失败案例: {failed_cases}")
    
    # 如果有成功的案例，打印NPV统计
    successful_results = df_results[df_results['Status'] == 'optimal']
    if not successful_results.empty:
        print(f"💰 NPV范围: {successful_results['NPV'].min():,.2f} 到 {successful_results['NPV'].max():,.2f}")
        print(f"📈 最优T值: T={successful_results.loc[successful_results['NPV'].idxmax(), 'T']}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_dir}/npv_vs_time_{timestamp}.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"💾 结果已保存到: {csv_filename}")
    
    return df_results, csv_filename

def main():
    """主函数"""
    print("NPV vs T 批量分析工具")
    print("=" * 60)
    
    # 运行批量分析
    df_results, csv_file = batch_solve_npv_analysis(t_min=1, t_max=50)
    
    # 简单统计输出
    print(f"\n📋 快速预览:")
    print(df_results.head(10))
    
    print(f"\n🎯 下一步:")
    print(f"   1. 查看完整结果: {csv_file}")
    print(f"   2. 运行可视化脚本: python optimal/visualization.py")

if __name__ == "__main__":
    main()
