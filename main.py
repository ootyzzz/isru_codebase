#!/usr/bin/env python3
"""
ISRU优化系统主程序
用于月球原位资源利用的优化决策支持系统
"""

import argparse
import json
import yaml
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from models.isru_model import ISRUOptimizationModel as ISRUModel
from solvers.glpk_solver import GLPKSolver
from analysis.gbm_demand import GBMDemandGenerator
from simulation.batch_runner import BatchRunner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查所有依赖项是否安装正确"""
    print("=== 依赖项检查 ===")
    
    # 检查Python版本
    import sys
    print(f"Python版本: {sys.version}")
    
    # 检查核心依赖
    dependencies = {
        'numpy': '数值计算',
        'pandas': '数据处理',
        'matplotlib': '可视化',
        'pyomo': '优化建模',
        'yaml': '配置文件处理',
        'scipy': '科学计算'
    }
    
    for dep, desc in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep} ({desc}) - 已安装")
        except ImportError:
            print(f"❌ {dep} ({desc}) - 未安装")
    
    # 检查GLPK求解器
    try:
        from pyomo.environ import SolverFactory
        glpk_available = SolverFactory('glpk').available()
        if glpk_available:
            print("✅ GLPK求解器 - 可用")
        else:
            print("❌ GLPK求解器 - 不可用")
    except Exception as e:
        print(f"❌ GLPK求解器检查失败: {e}")

def run_single_optimization(config_path='data/config.yaml', params_path='data/parameters.json'):
    """运行单次优化"""
    logger.info("开始单次优化运行...")
    
    try:
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        with open(params_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # 创建模型
        model = ISRUModel(params)
        
        # 构建模型
        T = params['economics']['T']
        demand_path = [0] + [params['demand']['D0'] * (1.02 ** t) for t in range(1, T + 1)]
        model.build_model(demand_path)
        
        # 求解
        results = model.solve()
        
        # 显示结果
        print("\n=== 优化结果 ===")
        print(f"状态: {results['status']}")
        print(f"目标值: {results['objective_value']}")
        
        if results['solution']:
            print("\n=== 决策变量 ===")
            for var_name, value in results['solution'].items():
                if value > 0:  # 只显示非零值
                    print(f"{var_name}: {value:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"单次优化运行失败: {e}")
        raise

def run_batch_simulation(scenario_type='parameter_sweep', num_scenarios=10):
    """运行批量仿真"""
    logger.info(f"开始批量仿真: {scenario_type}")
    
    try:
        runner = BatchRunner()
        results = runner.run_batch(
            scenario_type=scenario_type,
            num_scenarios=num_scenarios
        )
        
        print(f"\n=== 批量仿真完成 ===")
        print(f"运行场景数: {len(results)}")
        print(f"平均目标值: {sum(r['objective_value'] for r in results) / len(results):.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"批量仿真失败: {e}")
        raise

def run_sensitivity_analysis():
    """运行敏感性分析"""
    logger.info("开始敏感性分析...")
    
    try:
        runner = BatchRunner()
        results = runner.run_sensitivity_analysis()
        
        print("\n=== 敏感性分析结果 ===")
        for param, sensitivity in results.items():
            print(f"{param}: 敏感度 = {sensitivity['sensitivity']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"敏感性分析失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ISRU优化系统')
    parser.add_argument('--mode', choices=['check-deps', 'single', 'batch', 'sensitivity'],
                        default='single', help='运行模式')
    parser.add_argument('--config', default='data/config.yaml', help='配置文件路径')
    parser.add_argument('--params', default='data/parameters.json', help='参数文件路径')
    parser.add_argument('--scenario-type', default='parameter_sweep', 
                        choices=['parameter_sweep', 'monte_carlo', 'demand_scenarios'],
                        help='批量仿真场景类型')
    parser.add_argument('--num-scenarios', type=int, default=10, help='场景数量')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'check-deps':
            check_dependencies()
        
        elif args.mode == 'single':
            run_single_optimization(args.config, args.params)
        
        elif args.mode == 'batch':
            run_batch_simulation(args.scenario_type, args.num_scenarios)
        
        elif args.mode == 'sensitivity':
            run_sensitivity_analysis()
        
        else:
            print("未知模式，请使用 --help 查看可用选项")
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()