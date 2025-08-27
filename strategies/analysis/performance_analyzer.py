#!/usr/bin/env python3
"""
性能分析器
提供全面的策略性能分析和对比功能
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
from dataclasses import asdict
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.core.simulation_engine import SimulationResult
from strategies.utils.terminal_display import TerminalDisplay


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        初始化性能分析器
        
        Args:
            results_dir: 结果目录
        """
        self.results_dir = results_dir or Path(__file__).parent.parent / "simulation_results"
        
        # 设置matplotlib中文字体（如果可用）
        if HAS_PLOTTING:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 设置seaborn样式
            sns.set_style("whitegrid")
            sns.set_palette("husl")
    
    def analyze_financial_performance(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        分析财务性能
        
        Args:
            results: 仿真结果列表
            
        Returns:
            财务性能分析结果
        """
        if not results:
            return {}
        
        # 提取财务指标
        npvs = [r.performance_metrics.get('npv', 0) for r in results]
        revenues = [r.performance_metrics.get('total_revenue', 0) for r in results]
        costs = [r.performance_metrics.get('total_cost', 0) for r in results]
        
        # 计算财务统计
        financial_stats = {
            # NPV分析
            'npv_mean': float(np.mean(npvs)),
            'npv_median': float(np.median(npvs)),
            'npv_std': float(np.std(npvs)),
            'npv_min': float(np.min(npvs)),
            'npv_max': float(np.max(npvs)),
            'npv_q25': float(np.percentile(npvs, 25)),
            'npv_q75': float(np.percentile(npvs, 75)),
            'npv_skewness': float(self._calculate_skewness(npvs)),
            'npv_kurtosis': float(self._calculate_kurtosis(npvs)),
            
            # 风险指标
            'var_95': float(np.percentile(npvs, 5)),  # 95% VaR
            'cvar_95': float(np.mean([npv for npv in npvs if npv <= np.percentile(npvs, 5)])),  # 95% CVaR
            'probability_loss': float(np.mean([npv < 0 for npv in npvs])),
            'downside_deviation': float(self._calculate_downside_deviation(npvs)),
            
            # 收益成本分析
            'revenue_mean': float(np.mean(revenues)),
            'cost_mean': float(np.mean(costs)),
            'profit_margin_mean': float(np.mean([(r-c)/r for r, c in zip(revenues, costs) if r > 0])),
            
            # 稳定性指标
            'coefficient_of_variation': float(np.std(npvs) / np.mean(npvs)) if np.mean(npvs) != 0 else float('inf'),
            'sharpe_ratio': float(np.mean(npvs) / np.std(npvs)) if np.std(npvs) != 0 else 0
        }
        
        return financial_stats
    
    def analyze_operational_performance(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        分析运营性能
        
        Args:
            results: 仿真结果列表
            
        Returns:
            运营性能分析结果
        """
        if not results:
            return {}
        
        # 提取运营指标
        utilizations = [r.performance_metrics.get('avg_utilization', 0) for r in results]
        self_sufficiency_rates = [r.performance_metrics.get('self_sufficiency_rate', 0) for r in results]
        capacity_expansions = [r.performance_metrics.get('capacity_expansions', 0) for r in results]
        final_capacities = [r.performance_metrics.get('final_capacity', 0) for r in results]
        
        # 计算运营统计
        operational_stats = {
            # 利用率分析
            'utilization_mean': float(np.mean(utilizations)),
            'utilization_std': float(np.std(utilizations)),
            'utilization_min': float(np.min(utilizations)),
            'utilization_max': float(np.max(utilizations)),
            
            # 自给自足分析
            'self_sufficiency_mean': float(np.mean(self_sufficiency_rates)),
            'self_sufficiency_std': float(np.std(self_sufficiency_rates)),
            'full_self_sufficiency_rate': float(np.mean([rate >= 0.99 for rate in self_sufficiency_rates])),
            
            # 产能管理
            'avg_capacity_expansions': float(np.mean(capacity_expansions)),
            'final_capacity_mean': float(np.mean(final_capacities)),
            'final_capacity_std': float(np.mean(final_capacities)),
            
            # 效率指标
            'capacity_utilization_efficiency': float(np.mean([u for u in utilizations if u <= 1.0])),
            'expansion_frequency': float(np.mean([exp/len(r.decisions) for r, exp in zip(results, capacity_expansions) if r.decisions]))
        }
        
        return operational_stats
    
    def analyze_risk_profile(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        分析风险特征
        
        Args:
            results: 仿真结果列表
            
        Returns:
            风险特征分析结果
        """
        if not results:
            return {}
        
        npvs = [r.performance_metrics.get('npv', 0) for r in results]
        
        # 风险分析
        risk_stats = {
            # 基本风险指标
            'volatility': float(np.std(npvs)),
            'downside_risk': float(self._calculate_downside_deviation(npvs)),
            'upside_potential': float(self._calculate_upside_potential(npvs)),
            
            # 极值分析
            'max_drawdown': float(self._calculate_max_drawdown(npvs)),
            'tail_risk_5pct': float(np.percentile(npvs, 5)),
            'tail_risk_1pct': float(np.percentile(npvs, 1)),
            
            # 分布特征
            'skewness': float(self._calculate_skewness(npvs)),
            'kurtosis': float(self._calculate_kurtosis(npvs)),
            'is_normal_distribution': bool(self._test_normality(npvs)),
            
            # 风险调整收益
            'risk_adjusted_return': float(np.mean(npvs) / np.std(npvs)) if np.std(npvs) != 0 else 0,
            'sortino_ratio': float(self._calculate_sortino_ratio(npvs)),
            'calmar_ratio': float(self._calculate_calmar_ratio(npvs))
        }
        
        return risk_stats
    
    def compare_strategies(self, strategy_results: Dict[str, List[SimulationResult]]) -> Dict[str, Any]:
        """
        比较多个策略的性能
        
        Args:
            strategy_results: 策略结果字典
            
        Returns:
            策略比较分析结果
        """
        comparison = {}
        
        for strategy_name, results in strategy_results.items():
            comparison[strategy_name] = {
                'financial': self.analyze_financial_performance(results),
                'operational': self.analyze_operational_performance(results),
                'risk': self.analyze_risk_profile(results)
            }
        
        # 计算相对排名
        ranking = self._calculate_strategy_ranking(comparison)
        comparison['ranking'] = ranking
        
        return comparison
    
    def analyze_time_horizon_impact(self, horizon_results: Dict[int, Dict[str, List[SimulationResult]]]) -> Dict[str, Any]:
        """
        分析时间跨度对策略性能的影响
        
        Args:
            horizon_results: 时间跨度结果字典
            
        Returns:
            时间跨度影响分析结果
        """
        horizon_analysis = {}
        
        # 按策略分析时间跨度影响
        strategies = set()
        for horizon_data in horizon_results.values():
            strategies.update(horizon_data.keys())
        
        for strategy in strategies:
            strategy_horizon_data = {}
            
            for T, strategy_data in horizon_results.items():
                if strategy in strategy_data:
                    results = strategy_data[strategy]
                    npvs = [r.performance_metrics.get('npv', 0) for r in results]
                    
                    strategy_horizon_data[T] = {
                        'npv_mean': float(np.mean(npvs)),
                        'npv_std': float(np.std(npvs)),
                        'success_rate': float(np.mean([npv > 0 for npv in npvs]))
                    }
            
            horizon_analysis[strategy] = strategy_horizon_data
        
        # 计算时间跨度趋势
        trends = self._calculate_horizon_trends(horizon_analysis)
        horizon_analysis['trends'] = trends
        
        return horizon_analysis
    
    def generate_performance_report(self, strategy_results: Dict[str, List[SimulationResult]], 
                                  output_file: Optional[Path] = None) -> str:
        """
        生成性能分析报告
        
        Args:
            strategy_results: 策略结果字典
            output_file: 输出文件路径
            
        Returns:
            报告内容
        """
        report_lines = []
        
        # 报告标题
        report_lines.append("=" * 80)
        report_lines.append("ISRU策略性能分析报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 执行摘要
        comparison = self.compare_strategies(strategy_results)
        ranking = comparison.get('ranking', {})
        
        report_lines.append("📊 执行摘要")
        report_lines.append("-" * 40)
        if 'overall_ranking' in ranking:
            for i, (strategy, score) in enumerate(ranking['overall_ranking'], 1):
                report_lines.append(f"{i}. {strategy.title()}: 综合得分 {score:.2f}")
        report_lines.append("")
        
        # 详细分析
        for strategy_name, results in strategy_results.items():
            report_lines.append(f"🔍 {strategy_name.title()} 策略详细分析")
            report_lines.append("-" * 50)
            
            # 财务性能
            financial = self.analyze_financial_performance(results)
            report_lines.append("💰 财务性能:")
            report_lines.append(f"  NPV均值: ${financial.get('npv_mean', 0):,.0f}")
            report_lines.append(f"  NPV标准差: ${financial.get('npv_std', 0):,.0f}")
            report_lines.append(f"  盈利概率: {(1-financial.get('probability_loss', 0)):.1%}")
            report_lines.append(f"  夏普比率: {financial.get('sharpe_ratio', 0):.2f}")
            
            # 运营性能
            operational = self.analyze_operational_performance(results)
            report_lines.append("⚙️ 运营性能:")
            report_lines.append(f"  平均利用率: {operational.get('utilization_mean', 0):.1%}")
            report_lines.append(f"  自给自足率: {operational.get('self_sufficiency_mean', 0):.1%}")
            report_lines.append(f"  产能扩张次数: {operational.get('avg_capacity_expansions', 0):.1f}")
            
            # 风险特征
            risk = self.analyze_risk_profile(results)
            report_lines.append("⚠️ 风险特征:")
            report_lines.append(f"  波动率: ${risk.get('volatility', 0):,.0f}")
            report_lines.append(f"  下行风险: ${risk.get('downside_risk', 0):,.0f}")
            report_lines.append(f"  索提诺比率: {risk.get('sortino_ratio', 0):.2f}")
            
            report_lines.append("")
        
        # 策略建议
        report_lines.append("💡 策略建议")
        report_lines.append("-" * 40)
        report_lines.extend(self._generate_strategy_recommendations(comparison))
        
        # 生成报告文本
        report_text = "\n".join(report_lines)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skew = np.mean([((x - mean) / std) ** 3 for x in data])
        return skew
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurt = np.mean([((x - mean) / std) ** 4 for x in data]) - 3
        return kurt
    
    def _calculate_downside_deviation(self, data: List[float], target: float = 0) -> float:
        """计算下行偏差"""
        downside_returns = [min(0, x - target) for x in data]
        return np.sqrt(np.mean([x ** 2 for x in downside_returns]))
    
    def _calculate_upside_potential(self, data: List[float], target: float = 0) -> float:
        """计算上行潜力"""
        upside_returns = [max(0, x - target) for x in data]
        return np.mean(upside_returns)
    
    def _calculate_max_drawdown(self, data: List[float]) -> float:
        """计算最大回撤"""
        if not data:
            return 0.0
        
        cumulative = np.cumsum(data)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown))
    
    def _calculate_sortino_ratio(self, data: List[float], target: float = 0) -> float:
        """计算索提诺比率"""
        excess_return = np.mean(data) - target
        downside_dev = self._calculate_downside_deviation(data, target)
        
        if downside_dev == 0:
            return 0.0
        
        return excess_return / downside_dev
    
    def _calculate_calmar_ratio(self, data: List[float]) -> float:
        """计算卡尔玛比率"""
        annual_return = np.mean(data)
        max_drawdown = abs(self._calculate_max_drawdown(data))
        
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / max_drawdown
    
    def _test_normality(self, data: List[float]) -> bool:
        """测试数据是否符合正态分布"""
        try:
            from scipy import stats
            _, p_value = stats.normaltest(data)
            return p_value > 0.05  # 5%显著性水平
        except ImportError:
            # 如果没有scipy，使用简单的偏度和峰度检验
            skew = abs(self._calculate_skewness(data))
            kurt = abs(self._calculate_kurtosis(data))
            return skew < 2 and kurt < 7
    
    def _calculate_strategy_ranking(self, comparison: Dict[str, Dict]) -> Dict[str, Any]:
        """计算策略排名"""
        strategies = [k for k in comparison.keys() if k != 'ranking']
        
        # 定义评分权重
        weights = {
            'npv_mean': 0.3,
            'sharpe_ratio': 0.2,
            'probability_loss': -0.2,  # 负权重，损失概率越低越好
            'utilization_mean': 0.15,
            'self_sufficiency_mean': 0.15
        }
        
        # 计算综合得分
        scores = {}
        for strategy in strategies:
            score = 0
            for metric, weight in weights.items():
                if metric == 'probability_loss':
                    value = 1 - comparison[strategy]['financial'].get(metric, 0)
                elif metric in comparison[strategy]['financial']:
                    value = comparison[strategy]['financial'][metric]
                elif metric in comparison[strategy]['operational']:
                    value = comparison[strategy]['operational'][metric]
                else:
                    value = 0
                
                # 标准化值（简单线性缩放）
                normalized_value = min(max(value, 0), 1) if metric != 'npv_mean' else value / 1e6
                score += weight * normalized_value
            
            scores[strategy] = score
        
        # 排序
        overall_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'overall_ranking': overall_ranking,
            'scores': scores
        }
    
    def _calculate_horizon_trends(self, horizon_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """计算时间跨度趋势"""
        trends = {}
        
        for strategy, horizon_data in horizon_analysis.items():
            if strategy == 'trends':
                continue
            
            horizons = sorted(horizon_data.keys())
            npv_means = [horizon_data[h]['npv_mean'] for h in horizons]
            
            # 计算趋势斜率
            if len(horizons) >= 2:
                slope = np.polyfit(horizons, npv_means, 1)[0]
                trends[strategy] = {
                    'npv_trend_slope': float(slope),
                    'is_improving': slope > 0,
                    'trend_strength': float(abs(slope))
                }
        
        return trends
    
    def _generate_strategy_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """生成策略建议"""
        recommendations = []
        
        ranking = comparison.get('ranking', {})
        if 'overall_ranking' in ranking:
            best_strategy = ranking['overall_ranking'][0][0]
            recommendations.append(f"🏆 推荐策略: {best_strategy.title()}")
            recommendations.append(f"   该策略在综合评估中表现最佳")
            
            # 分析各策略特点
            for strategy_name, strategy_data in comparison.items():
                if strategy_name == 'ranking':
                    continue
                
                financial = strategy_data.get('financial', {})
                operational = strategy_data.get('operational', {})
                risk = strategy_data.get('risk', {})
                
                recommendations.append(f"")
                recommendations.append(f"📋 {strategy_name.title()} 策略特点:")
                
                # 财务特点
                if financial.get('npv_mean', 0) > 0:
                    recommendations.append(f"   ✓ 预期盈利: ${financial.get('npv_mean', 0):,.0f}")
                else:
                    recommendations.append(f"   ✗ 预期亏损: ${financial.get('npv_mean', 0):,.0f}")
                
                # 风险特点
                if risk.get('volatility', 0) < 1e6:
                    recommendations.append(f"   ✓ 低风险策略")
                else:
                    recommendations.append(f"   ⚠️ 高风险策略")
                
                # 运营特点
                if operational.get('utilization_mean', 0) > 0.8:
                    recommendations.append(f"   ✓ 高效利用产能")
                
                if operational.get('self_sufficiency_mean', 0) > 0.9:
                    recommendations.append(f"   ✓ 高自给自足率")
        
        return recommendations


if __name__ == "__main__":
    # 测试代码
    print("=== 性能分析器测试 ===")
    
    # 这里需要实际的仿真结果来测试
    # 由于测试环境限制，这里只做基本功能测试
    
    analyzer = PerformanceAnalyzer()
    
    # 测试基本统计函数
    test_data = [100, 150, 120, 180, 200, 90, 160, 140, 170, 130]
    
    print(f"偏度: {analyzer._calculate_skewness(test_data):.3f}")
    print(f"峰度: {analyzer._calculate_kurtosis(test_data):.3f}")
    print(f"下行偏差: {analyzer._calculate_downside_deviation(test_data):.3f}")
    print(f"索提诺比率: {analyzer._calculate_sortino_ratio(test_data):.3f}")
    print(f"最大回撤: {analyzer._calculate_max_drawdown(test_data):.3f}")