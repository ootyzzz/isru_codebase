"""
测试 ISRU 优化模型 (5年期，GBM随机需求)
"""

import json
import numpy as np
from models.isru_model import create_and_solve_model
from analysis.gbm_demand import GBMDemandGenerator

def main():
    # 加载参数
    with open("data/parameters.json", "r") as f:
        params = json.load(f)

    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 使用GBM生成随机需求路径
    T = params["economics"]["T"]
    expected_length = T + 1
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
        print(f"调整需求路径长度: {len(demand_path)} (期望: {T})")

    # 运行优化
    result = create_and_solve_model(params, demand_path, solver_name="glpk")

    # 打印结果
    if result["status"] == "optimal":
        print("✅ 模型求解成功！")
        print(f"最优目标值 (NPV): {result['objective_value']:,.2f}")
        
        # 打印结果摘要
        from models.isru_model import ISRUOptimizationModel
        model = ISRUOptimizationModel(params)
        model.solution = result["solution"]
        model.demand_path = demand_path
        model.print_solution_summary()
    else:
        print(f"❌ 求解失败: {result['status']}")
        if "termination_condition" in result:
            print(f"原因: {result['termination_condition']}")
        if "error" in result:
            print(f"错误: {result['error']}")

if __name__ == "__main__":
    main()
