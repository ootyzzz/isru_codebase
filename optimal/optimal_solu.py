"""
测试 ISRU 优化模型 (t年期，GBM随机需求)
"""

import json
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.isru_model import create_and_solve_model
from analysis.gbm_demand import GBMDemandGenerator

def solve_isru_optimization(T_years, params_file="data/parameters.json", random_seed=42, verbose=True):
    """
    求解ISRU优化问题的核心函数
    
    Args:
        T_years: 时间长度（年）
        params_file: 参数文件路径
        random_seed: 随机种子
        verbose: 是否打印详细信息
        
    Returns:
        dict: 包含求解结果的字典
    """
    # 加载参数
    with open(params_file, "r") as f:
        params = json.load(f)
    
    # 修改时间参数
    params["economics"]["T"] = T_years
    
    # 设置随机种子以确保结果可重现
    np.random.seed(random_seed)
    
    # 使用GBM生成随机需求路径
    expected_length = T_years + 1
    demand_params = params['demand']
    gbm_generator = GBMDemandGenerator(
        D0=demand_params['D0'],
        mu=demand_params['mu'],
        sigma=demand_params['sigma'],
        dt=demand_params['dt']
    )
    demand_path = gbm_generator.generate_single_path(expected_length)
    
    # 验证需求路径长度 - GBM生成器返回T+1个点（包括t=0）
    if len(demand_path) != expected_length:
        # 调整需求路径，去掉t=0的初始点
        demand_path = demand_path[1:]  # 只保留t=1到t=T
        if verbose:
            print(f"调整需求路径长度: {len(demand_path)} (期望: {T_years})")
    
    # 确保 demand_path 是列表而不是numpy数组
    if hasattr(demand_path, 'tolist'):
        demand_path = demand_path.tolist()

    # 运行优化
    result = create_and_solve_model(params, demand_path, solver_name="glpk")
    
    # 构造返回结果
    output = {
        "T": T_years,
        "status": result["status"],
        "npv": result.get("objective_value", None),
        "solve_time": result.get("solve_time", None),
        "demand_path": demand_path,
        "params": params,
        "full_result": result
    }
    
    # 打印结果（如果需要）
    if verbose:
        if result["status"] == "optimal":
            print(f"✅ T={T_years}: NPV = {result['objective_value']:,.2f}")
        else:
            print(f"❌ T={T_years}: 求解失败 - {result['status']}")
            if "termination_condition" in result:
                print(f"   原因: {result['termination_condition']}")
    
    return output

def main():
    """运行单次优化（保持原有功能）"""
    # 加载参数获取默认T值
    with open("data/parameters.json", "r") as f:
        params = json.load(f)
    
    T = params["economics"]["T"]
    result = solve_isru_optimization(T, verbose=True)
    
    # 打印详细结果摘要
    if result["status"] == "optimal":
        print("\n=== 详细结果摘要 ===")
        from models.isru_model import ISRUOptimizationModel
        model = ISRUOptimizationModel(result["params"])
        model.solution = result["full_result"]["solution"]
        model.demand_path = result["demand_path"]
        model.print_solution_summary()
    else:
        if "termination_condition" in result["full_result"]:
            print(f"原因: {result['full_result']['termination_condition']}")
        if "error" in result["full_result"]:
            print(f"错误: {result['full_result']['error']}")

if __name__ == "__main__":
    main()
