"""
GLPK求解器接口
提供与GLPK求解器的集成接口
"""

import logging
import os
import subprocess
from typing import Dict, Any, Optional
from pyomo.environ import SolverFactory
from pyomo.opt import SolverResults

logger = logging.getLogger(__name__)


class GLPKSolver:
    """
    GLPK求解器包装类
    
    提供对GLPK求解器的完整接口，包括求解器配置、
    选项设置和结果处理。
    """
    
    def __init__(self, solver_path: Optional[str] = None):
        """
        初始化GLPK求解器
        
        Args:
            solver_path: GLPK求解器可执行文件路径（可选）
        """
        self.solver_path = solver_path
        self.solver = None
        self._setup_solver()
    
    def _setup_solver(self) -> None:
        """设置GLPK求解器"""
        try:
            # 尝试创建求解器实例
            self.solver = SolverFactory('glpk')
            
            if self.solver is None or not self.solver.available():
                raise RuntimeError("GLPK求解器不可用")
            
            logger.info("GLPK求解器初始化成功")
            
        except Exception as e:
            logger.error(f"GLPK求解器初始化失败: {e}")
            # 尝试安装GLPK
            self._install_glpk()
    
    def _install_glpk(self) -> None:
        """尝试安装GLPK求解器"""
        logger.info("尝试安装GLPK求解器...")
        
        try:
            # 检查conda是否可用
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("使用conda安装glpk...")
                subprocess.run(['conda', 'install', '-y', 'glpk'], check=True)
                logger.info("GLPK安装成功")
                self.solver = SolverFactory('glpk')
            else:
                logger.warning("conda不可用，请手动安装GLPK")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"安装GLPK失败: {e}")
            logger.info("请手动安装GLPK: conda install glpk")
    
    def solve(self, model, **kwargs) -> Dict[str, Any]:
        """
        求解优化模型
        
        Args:
            model: Pyomo模型
            **kwargs: 求解器选项
            
        Returns:
            求解结果字典
        """
        if self.solver is None:
            raise RuntimeError("GLPK求解器未初始化")
        
        # 默认求解器选项
        default_options = {
            'tmlim': 3600,  # 时间限制（秒）
            'mipgap': 0.01,  # MIP间隙
            'msg_lev': 3,   # 消息级别
            'presolve': True,
            'scale': True
        }
        
        # 合并用户选项
        solver_options = {**default_options, **kwargs}
        
        try:
            # 设置求解器选项
            for key, value in solver_options.items():
                self.solver.options[key] = value
            
            # 求解模型
            logger.info("开始求解模型...")
            results = self.solver.solve(model, tee=True)
            
            # 处理结果
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"求解过程中出错: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _process_results(self, results: SolverResults) -> Dict[str, Any]:
        """处理求解结果"""
        status_map = {
            'optimal': 'optimal',
            'infeasible': 'infeasible',
            'unbounded': 'unbounded',
            'limit': 'limit',
            'error': 'error',
            'unknown': 'unknown'
        }
        
        solver_status = str(results.solver.termination_condition)
        status = status_map.get(solver_status, 'unknown')
        
        result = {
            'status': status,
            'solver_status': solver_status,
            'solve_time': results.solver.time,
            'iterations': results.solver.iterations,
            'objective_value': None
        }
        
        if status == 'optimal':
            # 提取目标值
            from pyomo.environ import value
            if hasattr(results.problem, 'objective'):
                result['objective_value'] = value(results.problem.objective)
        
        return result
    
    def get_solver_info(self) -> Dict[str, Any]:
        """获取求解器信息"""
        if self.solver is None:
            return {'available': False}
        
        return {
            'available': True,
            'name': 'GLPK',
            'version': self.solver.version(),
            'executable': self.solver.executable()
        }


class SolverConfig:
    """求解器配置类"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """获取默认求解器配置"""
        return {
            'solver_name': 'glpk',
            'options': {
                'tmlim': 3600,
                'mipgap': 0.01,
                'msg_lev': 3,
                'presolve': True,
                'scale': True
            }
        }
    
    @staticmethod
    def get_fast_config() -> Dict[str, Any]:
        """获取快速求解配置"""
        return {
            'solver_name': 'glpk',
            'options': {
                'tmlim': 300,
                'mipgap': 0.05,
                'msg_lev': 2,
                'presolve': True,
                'scale': True
            }
        }
    
    @staticmethod
    def get_precise_config() -> Dict[str, Any]:
        """获取精确求解配置"""
        return {
            'solver_name': 'glpk',
            'options': {
                'tmlim': 7200,
                'mipgap': 0.001,
                'msg_lev': 4,
                'presolve': True,
                'scale': True,
                'cuts': True
            }
        }


def check_glpk_availability() -> bool:
    """检查GLPK是否可用"""
    try:
        solver = SolverFactory('glpk')
        return solver.available()
    except:
        return False


def get_glpk_installation_guide() -> str:
    """获取GLPK安装指南"""
    return """
    GLPK安装指南:
    
    1. 使用conda安装（推荐）:
       conda install glpk
    
    2. 使用pip安装:
       pip install glpk
    
    3. 手动安装:
       - Windows: 下载GLPK for Windows
       - Linux: sudo apt-get install glpk-utils
       - macOS: brew install glpk
    
    4. 验证安装:
       python -c "from pyomo.environ import SolverFactory; print(SolverFactory('glpk').available())"
    """


if __name__ == "__main__":
    # 测试GLPK求解器
    print("检查GLPK可用性...")
    
    if check_glpk_availability():
        print("✓ GLPK求解器可用")
        
        solver = GLPKSolver()
        info = solver.get_solver_info()
        print(f"求解器信息: {info}")
    else:
        print("✗ GLPK求解器不可用")
        print(get_glpk_installation_guide())