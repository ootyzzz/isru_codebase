#!/usr/bin/env python3
"""
ISRU 模型基本使用示例
展示如何加载参数、生成需求路径并求解模型
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from models.isru_model import create_and_solve_model
from analysis.gbm_demand import GBMDemandGenerator

def main():
    """运行基本示例"""
    print("=== ISRU 模型基本示例 ===\n")
    
    # 1. 加载参数
    print("1. 加载参数...")
    with open("data/parameters.json") as f:
        params = json.load(f)
    
    # 打印关键参数
    T = params['economics']['T']
    print(f"   时间周期: {T} 个月")
    print(f"   氧气售价: ${params['economics']['P_m']:,}")
    print(f"   初始需求: {params['demand']['D0']} kg")
    
    # 2. 创建需求生成器
    print("\n2. 创建GBM需求生成器...")
    demand_params = params['demand']
    gbm_generator = GBMDemandGenerator(
        D0=demand_params['D0'],
        mu=demand_params['mu'],
        sigma=demand_params['sigma'],
        dt=demand_params['dt']
    )
    
    # 3. 生成需求路径
    print("\n3. 生成需求路径...")
    np.random.seed(42)  # 确保结果可重现
    demand_path = gbm_generator.generate_single_path(T)
    
    print(f"   需求路径长度: {len(demand_path)}")
    print(f"   初始需求: {demand_path[0]:.2f} kg")
    print(f"   最终需求: {demand_path[-1]:.2f} kg")
    print(f"   平均需求: {np.mean(demand_path):.2f} kg")
    
    # 4. 求解模型
    print("\n4. 求解优化模型...")
    result = create_and_solve_model(params, demand_path, solver_name="glpk")
    
    if result['status'] == 'optimal':
        print("   ✅ 求解成功!")
        print(f"   NPV: ${result['objective_value']:,.2f}")
        
        # 提取关键决策变量
        solution = result['solution']
        Mt_total = sum(solution['delta_Mt'].values())
        print(f"   总开采量: {Mt_total:.2f} kg")
        
        # 打印各期决策
        print("\n5. 各期决策:")
        for t in range(T):
            print(f"   月份 {t+1:2d}: "
                  f"开采 {solution['delta_Mt'][t]:6.2f} kg, "
                  f"库存 {solution['St'][t]:6.2f} kg, "
                  f"需求 {demand_path[t+1]:6.2f} kg")
        
        # 保存结果
        with open('data/example_results.json', 'w') as f:
            json.dump({
                'parameters': params,
                'demand_path': demand_path.tolist(),
                'solution': solution,
                'npv': result['objective_value']
            }, f, indent=2)
        print("\n6. 结果已保存到 data/example_results.json")
        
    else:
        print(f"   ❌ 求解失败: {result['status']}")

if __name__ == "__main__":
    main()