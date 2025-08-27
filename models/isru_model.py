"""
ISRU氧气生产优化模型
整合所有模型组件的完整优化模型
"""

import logging
from pyomo.environ import ConcreteModel, value as pyo_value
from typing import Dict, Any, Optional
import pandas as pd

from .variables import define_variables
from .constraints import define_constraints, validate_constraints
from .objective import define_objective, calculate_detailed_costs

logger = logging.getLogger(__name__)


class ISRUOptimizationModel:
    """
    ISRU氧气生产优化模型
    
    这是一个完整的Pyomo优化模型，用于优化月球氧气生产系统的部署和运营。
    模型考虑了需求不确定性、技术约束和成本因素。
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        初始化优化模型
        
        Args:
            params: 参数字典，包含所有模型参数
        """
        self.params = params
        self.model = None
        self.demand_path = None
        self.solution = None
        
    def build_model(self, demand_path: list) -> ConcreteModel:
        """
        构建完整的优化模型
        
        Args:
            demand_path: 需求路径列表
            
        Returns:
            完整的Pyomo模型
        """
        self.demand_path = demand_path
        
        # 验证需求路径长度
        T = self.params['economics']['T']
        expected_length = T + 1  # 包括时间0到T
        
        if len(demand_path) != expected_length:
            raise ValueError(
                f"需求路径长度错误: 期望 {expected_length} (T={T} + 1), "
                f"实际 {len(demand_path)}"
            )
        
        # 创建模型
        self.model = ConcreteModel(name="ISRU_Oxygen_Optimization")
        
        # 定义变量
        logger.info("定义决策变量...")
        self.model = define_variables(self.model, self.params)
        
        # 定义约束
        logger.info("定义约束条件...")
        self.model = define_constraints(self.model, self.params, demand_path)
        
        # 定义目标函数
        logger.info("定义目标函数...")
        self.model = define_objective(self.model, self.params)
        
        logger.info("模型构建完成")
        return self.model
    
    def solve(self, solver_name: str = 'glpk', solver_options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        求解优化模型
        
        Args:
            solver_name: 求解器名称
            solver_options: 求解器选项
            
        Returns:
            求解结果字典
        """
        if self.model is None:
            raise ValueError("模型尚未构建，请先调用build_model()")
        
        try:
            # 导入求解器
            from pyomo.opt import SolverFactory
            
            # 创建求解器
            solver = SolverFactory(solver_name)
            
            if solver_options:
                for key, opt_val in solver_options.items():   # 避免使用 value
                    solver.options[key] = opt_val
            
            # 求解
            logger.info(f"使用{solver_name}求解器求解模型...")
            results = solver.solve(self.model, tee=True)
            
            # 检查求解状态
            if str(results.solver.termination_condition).lower() == 'optimal':
                logger.info("模型求解成功")
                
                # 验证约束
                if validate_constraints(self.model):
                    logger.info("所有约束验证通过")
                else:
                    logger.warning("存在约束违反")
                
                # 提取解
                self.solution = self._extract_solution()
                
                # 计算详细成本
                self.solution['costs'] = calculate_detailed_costs(self.model, self.params)
                
                return {
                    'status': 'optimal',
                    'objective_value': pyo_value(self.model.NPV),
                    'solution': self.solution,
                    'solver_results': results
                }
            else:
                logger.error(f"求解失败: {results.solver.termination_condition}")
                return {
                    'status': 'failed',
                    'termination_condition': str(results.solver.termination_condition),
                    'solver_results': results
                }
                
        except Exception as e:
            logger.error(f"求解过程中出错: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _extract_solution(self) -> Dict[str, Any]:
        """提取求解结果"""
        if self.model is None:
            return {}
        
        solution = {
            'Qt': {t: pyo_value(self.model.Qt[t]) for t in self.model.T},
            'Qt_cap': {t: pyo_value(self.model.Qt_cap[t]) for t in self.model.T},
            'St': {t: pyo_value(self.model.St[t]) for t in self.model.T},
            'Et': {t: pyo_value(self.model.Et[t]) for t in self.model.T},
            'Mt': {t: pyo_value(self.model.Mt[t]) for t in self.model.T},
            'delta_Mt': {t: pyo_value(self.model.delta_Mt[t]) for t in self.model.T},
            'M_leo': {t: pyo_value(self.model.M_leo[t]) for t in self.model.T},
            'NPV': pyo_value(self.model.NPV)
        }
        
        return solution
    
    def get_solution_dataframe(self) -> pd.DataFrame:
        """将解转换为DataFrame格式"""
        if self.solution is None:
            raise ValueError("模型尚未求解，请先调用solve()")
        
        # 创建DataFrame
        df = pd.DataFrame({
            'time': list(self.solution['Qt'].keys()),
            'demand': self.demand_path,
            'Qt': list(self.solution['Qt'].values()),
            'Qt_cap': list(self.solution['Qt_cap'].values()),
            'St': list(self.solution['St'].values()),
            'Et': list(self.solution['Et'].values()),
            'Mt': list(self.solution['Mt'].values()),
            'delta_Mt': list(self.solution['delta_Mt'].values()),
            'M_leo': list(self.solution['M_leo'].values())
        })
        
        return df
    
    def save_solution(self, filepath: str) -> None:
        """保存求解结果到文件"""
        if self.solution is None:
            raise ValueError("模型尚未求解，请先调用solve()")
        
        df = self.get_solution_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"求解结果已保存到: {filepath}")
    
    def print_solution_summary(self) -> None:
        """打印求解结果摘要"""
        if self.solution is None:
            print("模型尚未求解")
            return
        
        print("\n=== 求解结果摘要 ===")
        print(f"净现值 (NPV): ${self.solution['NPV']:,.2f}")
        print(f"总交付氧气: {sum(self.solution['Qt'].values()):,.2f} kg")
        print(f"总短缺: {sum(self.solution['St'].values()):,.2f} kg")
        print(f"总剩余: {sum(self.solution['Et'].values()):,.2f} kg")
        print(f"最大ISRU质量: {max(self.solution['Mt'].values()):,.2f} kg")
        print(f"总新增部署: {sum(self.solution['delta_Mt'].values()):,.2f} kg")
        
        if 'costs' in self.solution:
            from .objective import print_cost_breakdown
            print_cost_breakdown(self.solution['costs'])


# 便捷函数
def create_and_solve_model(params: Dict[str, Any], demand_path: list, 
                          solver_name: str = 'glpk') -> Dict[str, Any]:
    """
    便捷函数：创建并求解模型
    
    Args:
        params: 参数字典
        demand_path: 需求路径
        solver_name: 求解器名称
        
    Returns:
        求解结果
    """
    model = ISRUOptimizationModel(params)
    model.build_model(demand_path)
    return model.solve(solver_name)


if __name__ == "__main__":
    # 测试代码
    import json
    
    # 加载参数
    with open('data/parameters.json', 'r') as f:
        params = json.load(f)
    
    # 创建需求路径（使用平均需求）
    T = params['economics']['T']
    D0 = params['demand']['D0']
    demand_path = [0] + [D0 * (1.02 ** t) for t in range(1, T + 1)]
    
    # 创建并求解模型
    result = create_and_solve_model(params, demand_path)
    
    if result['status'] == 'optimal':
        print("模型求解成功！")
        print(f"最优目标值: ${result['objective_value']:,.2f}")
    else:
        print(f"求解失败: {result['status']}")
