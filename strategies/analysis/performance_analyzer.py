#!/usr/bin/env python3
"""
Performance Analyzer
Provide comprehensive strategy performance analysis and comparison functionality
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
    """Performance analyzer"""
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize performance analyzer
        
        Args:
            results_dir: Results directory
        """
        self.results_dir = results_dir or Path(__file__).parent.parent / "simulation_results"
        
        # Set matplotlib fonts (if available)
        if HAS_PLOTTING:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            
            # Set seaborn style
            sns.set_style("whitegrid")
            sns.set_palette("husl")
    
    def analyze_financial_performance(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Analyze financial performance
        
        Args:
            results: Simulation results list
            
        Returns:
            Financial performance analysis results
        """
        if not results:
            return {}
        
        # Extract financial metrics
        npvs = [r.performance_metrics.get('npv', 0) for r in results]
        revenues = [r.performance_metrics.get('total_revenue', 0) for r in results]
        costs = [r.performance_metrics.get('total_cost', 0) for r in results]
        
        # Calculate financial statistics
        financial_stats = {
            # NPV analysis
            'npv_mean': float(np.mean(npvs)),
            'npv_median': float(np.median(npvs)),
            'npv_std': float(np.std(npvs)),
            'npv_min': float(np.min(npvs)),
            'npv_max': float(np.max(npvs)),
            'npv_q25': float(np.percentile(npvs, 25)),
            'npv_q75': float(np.percentile(npvs, 75)),
            'npv_skewness': float(self._calculate_skewness(npvs)),
            'npv_kurtosis': float(self._calculate_kurtosis(npvs)),
            
            # Risk indicators
            'var_95': float(np.percentile(npvs, 5)),  # 95% VaR
            'cvar_95': float(np.mean([npv for npv in npvs if npv <= np.percentile(npvs, 5)])),  # 95% CVaR
            'probability_loss': float(np.mean([npv < 0 for npv in npvs])),
            'downside_deviation': float(self._calculate_downside_deviation(npvs)),
            
            # Revenue cost analysis
            'revenue_mean': float(np.mean(revenues)),
            'cost_mean': float(np.mean(costs)),
            'profit_margin_mean': float(np.mean([(r-c)/r for r, c in zip(revenues, costs) if r > 0])),
            
            # Stability indicators
            'coefficient_of_variation': float(np.std(npvs) / np.mean(npvs)) if np.mean(npvs) != 0 else float('inf'),
            'sharpe_ratio': float(np.mean(npvs) / np.std(npvs)) if np.std(npvs) != 0 else 0
        }
        
        return financial_stats
    
    def analyze_operational_performance(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Analyze operational performance
        
        Args:
            results: Simulation results list
            
        Returns:
            Operational performance analysis results
        """
        if not results:
            return {}
        
        # Extract operational metrics
        utilizations = [r.performance_metrics.get('avg_utilization', 0) for r in results]
        self_sufficiency_rates = [r.performance_metrics.get('self_sufficiency_rate', 0) for r in results]
        capacity_expansions = [r.performance_metrics.get('capacity_expansions', 0) for r in results]
        final_capacities = [r.performance_metrics.get('final_capacity', 0) for r in results]
        
        # Calculate operational statistics
        operational_stats = {
            # Utilization analysis
            'utilization_mean': float(np.mean(utilizations)),
            'utilization_std': float(np.std(utilizations)),
            'utilization_min': float(np.min(utilizations)),
            'utilization_max': float(np.max(utilizations)),
            
            # Self-sufficiency analysis
            'self_sufficiency_mean': float(np.mean(self_sufficiency_rates)),
            'self_sufficiency_std': float(np.std(self_sufficiency_rates)),
            'full_self_sufficiency_rate': float(np.mean([rate >= 0.99 for rate in self_sufficiency_rates])),
            
            # Capacity management
            'avg_capacity_expansions': float(np.mean(capacity_expansions)),
            'final_capacity_mean': float(np.mean(final_capacities)),
            'final_capacity_std': float(np.mean(final_capacities)),
            
            # Efficiency indicators
            'capacity_utilization_efficiency': float(np.mean([u for u in utilizations if u <= 1.0])),
            'expansion_frequency': float(np.mean([exp/len(r.decisions) for r, exp in zip(results, capacity_expansions) if r.decisions]))
        }
        
        return operational_stats
    
    def analyze_risk_profile(self, results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Analyze risk profile
        
        Args:
            results: Simulation results list
            
        Returns:
            Risk profile analysis results
        """
        if not results:
            return {}
        
        npvs = [r.performance_metrics.get('npv', 0) for r in results]
        
        # Risk analysis
        risk_stats = {
            # Basic risk indicators
            'volatility': float(np.std(npvs)),
            'downside_risk': float(self._calculate_downside_deviation(npvs)),
            'upside_potential': float(self._calculate_upside_potential(npvs)),
            
            # Extreme value analysis
            'max_drawdown': float(self._calculate_max_drawdown(npvs)),
            'tail_risk_5pct': float(np.percentile(npvs, 5)),
            'tail_risk_1pct': float(np.percentile(npvs, 1)),
            
            # Distribution characteristics
            'skewness': float(self._calculate_skewness(npvs)),
            'kurtosis': float(self._calculate_kurtosis(npvs)),
            'is_normal_distribution': bool(self._test_normality(npvs)),
            
            # Risk-adjusted returns
            'risk_adjusted_return': float(np.mean(npvs) / np.std(npvs)) if np.std(npvs) != 0 else 0,
            'sortino_ratio': float(self._calculate_sortino_ratio(npvs)),
            'calmar_ratio': float(self._calculate_calmar_ratio(npvs))
        }
        
        return risk_stats
    
    def compare_strategies(self, strategy_results: Dict[str, List[SimulationResult]]) -> Dict[str, Any]:
        """
        Compare performance of multiple strategies
        
        Args:
            strategy_results: Strategy results dictionary
            
        Returns:
            Strategy comparison analysis results
        """
        comparison = {}
        
        for strategy_name, results in strategy_results.items():
            comparison[strategy_name] = {
                'financial': self.analyze_financial_performance(results),
                'operational': self.analyze_operational_performance(results),
                'risk': self.analyze_risk_profile(results)
            }
        
        # Calculate relative ranking
        ranking = self._calculate_strategy_ranking(comparison)
        comparison['ranking'] = ranking
        
        return comparison
    
    def analyze_time_horizon_impact(self, horizon_results: Dict[int, Dict[str, List[SimulationResult]]]) -> Dict[str, Any]:
        """
        Analyze impact of time horizon on strategy performance
        
        Args:
            horizon_results: Time horizon results dictionary
            
        Returns:
            Time horizon impact analysis results
        """
        horizon_analysis = {}
        
        # Analyze time horizon impact by strategy
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
        
        # Calculate time horizon trends
        trends = self._calculate_horizon_trends(horizon_analysis)
        horizon_analysis['trends'] = trends
        
        return horizon_analysis
    
    def generate_performance_report(self, strategy_results: Dict[str, List[SimulationResult]], 
                                  output_file: Optional[Path] = None) -> str:
        """
        Generate performance analysis report
        
        Args:
            strategy_results: Strategy results dictionary
            output_file: Output file path
            
        Returns:
            Report content
        """
        report_lines = []
        
        # Report title
        report_lines.append("=" * 80)
        report_lines.append("ISRU Strategy Performance Analysis Report")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Executive summary
        comparison = self.compare_strategies(strategy_results)
        ranking = comparison.get('ranking', {})
        
        report_lines.append("Executive Summary")
        report_lines.append("-" * 40)
        if 'overall_ranking' in ranking:
            for i, (strategy, score) in enumerate(ranking['overall_ranking'], 1):
                report_lines.append(f"{i}. {strategy.title()}: Overall Score {score:.2f}")
        report_lines.append("")
        
        # Detailed analysis
        for strategy_name, results in strategy_results.items():
            report_lines.append(f"[Analysis] {strategy_name.title()} Strategy Detailed Analysis")
            report_lines.append("-" * 50)
            
            # Financial performance
            financial = self.analyze_financial_performance(results)
            report_lines.append("Financial Performance:")
            report_lines.append(f"  NPV Mean: ${financial.get('npv_mean', 0):,.0f}")
            report_lines.append(f"  NPV Std Dev: ${financial.get('npv_std', 0):,.0f}")
            report_lines.append(f"  Profit Probability: {(1-financial.get('probability_loss', 0)):.1%}")
            report_lines.append(f"  Sharpe Ratio: {financial.get('sharpe_ratio', 0):.2f}")
            
            # Operational performance
            operational = self.analyze_operational_performance(results)
            report_lines.append("Operational Performance:")
            report_lines.append(f"  Average Utilization: {operational.get('utilization_mean', 0):.1%}")
            report_lines.append(f"  Self-Sufficiency Rate: {operational.get('self_sufficiency_mean', 0):.1%}")
            report_lines.append(f"  Capacity Expansions: {operational.get('avg_capacity_expansions', 0):.1f}")
            
            # Risk characteristics
            risk = self.analyze_risk_profile(results)
            report_lines.append("Risk Characteristics:")
            report_lines.append(f"  Volatility: ${risk.get('volatility', 0):,.0f}")
            report_lines.append(f"  Downside Risk: ${risk.get('downside_risk', 0):,.0f}")
            report_lines.append(f"  Sortino Ratio: {risk.get('sortino_ratio', 0):.2f}")
            
            report_lines.append("")
        
        # Strategy recommendations
        report_lines.append("Strategy Recommendations")
        report_lines.append("-" * 40)
        report_lines.extend(self._generate_strategy_recommendations(comparison))
        
        # Generate report text
        report_text = "\n".join(report_lines)
        
        # Save report
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        skew = np.mean([((x - mean) / std) ** 3 for x in data])
        return skew
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        kurt = np.mean([((x - mean) / std) ** 4 for x in data]) - 3
        return kurt
    
    def _calculate_downside_deviation(self, data: List[float], target: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = [min(0, x - target) for x in data]
        return np.sqrt(np.mean([x ** 2 for x in downside_returns]))
    
    def _calculate_upside_potential(self, data: List[float], target: float = 0) -> float:
        """Calculate upside potential"""
        upside_returns = [max(0, x - target) for x in data]
        return np.mean(upside_returns)
    
    def _calculate_max_drawdown(self, data: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not data:
            return 0.0
        
        cumulative = np.cumsum(data)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return float(np.min(drawdown))
    
    def _calculate_sortino_ratio(self, data: List[float], target: float = 0) -> float:
        """Calculate Sortino ratio"""
        excess_return = np.mean(data) - target
        downside_dev = self._calculate_downside_deviation(data, target)
        
        if downside_dev == 0:
            return 0.0
        
        return excess_return / downside_dev
    
    def _calculate_calmar_ratio(self, data: List[float]) -> float:
        """Calculate Calmar ratio"""
        annual_return = np.mean(data)
        max_drawdown = abs(self._calculate_max_drawdown(data))
        
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / max_drawdown
    
    def _test_normality(self, data: List[float]) -> bool:
        """Test whether data follows normal distribution"""
        try:
            from scipy import stats
            _, p_value = stats.normaltest(data)
            return p_value > 0.05  # 5% significance level
        except ImportError:
            # If scipy is not available, use simple skewness and kurtosis test
            skew = abs(self._calculate_skewness(data))
            kurt = abs(self._calculate_kurtosis(data))
            return skew < 2 and kurt < 7
    
    def _calculate_strategy_ranking(self, comparison: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate strategy ranking"""
        strategies = [k for k in comparison.keys() if k != 'ranking']
        
        # Define scoring weights
        weights = {
            'npv_mean': 0.3,
            'sharpe_ratio': 0.2,
            'probability_loss': -0.2,  # Negative weight, lower loss probability is better
            'utilization_mean': 0.15,
            'self_sufficiency_mean': 0.15
        }
        
        # Calculate comprehensive score
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
                
                # Normalize value (simple linear scaling)
                normalized_value = min(max(value, 0), 1) if metric != 'npv_mean' else value / 1e6
                score += weight * normalized_value
            
            scores[strategy] = score
        
        # Sort
        overall_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'overall_ranking': overall_ranking,
            'scores': scores
        }
    
    def _calculate_horizon_trends(self, horizon_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate time horizon trends"""
        trends = {}
        
        for strategy, horizon_data in horizon_analysis.items():
            if strategy == 'trends':
                continue
            
            horizons = sorted(horizon_data.keys())
            npv_means = [horizon_data[h]['npv_mean'] for h in horizons]
            
            # Calculate trend slope
            if len(horizons) >= 2:
                slope = np.polyfit(horizons, npv_means, 1)[0]
                trends[strategy] = {
                    'npv_trend_slope': float(slope),
                    'is_improving': slope > 0,
                    'trend_strength': float(abs(slope))
                }
        
        return trends
    
    def _generate_strategy_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate strategy recommendations"""
        recommendations = []
        
        ranking = comparison.get('ranking', {})
        if 'overall_ranking' in ranking:
            best_strategy = ranking['overall_ranking'][0][0]
            recommendations.append(f"Recommended Strategy: {best_strategy.title()}")
            recommendations.append(f"   This strategy performs best in comprehensive evaluation")
            
            # Analyze characteristics of each strategy
            for strategy_name, strategy_data in comparison.items():
                if strategy_name == 'ranking':
                    continue
                
                financial = strategy_data.get('financial', {})
                operational = strategy_data.get('operational', {})
                risk = strategy_data.get('risk', {})
                
                recommendations.append(f"")
                recommendations.append(f"{strategy_name.title()} Strategy Characteristics:")
                
                # Financial characteristics
                if financial.get('npv_mean', 0) > 0:
                    recommendations.append(f"   + Expected Profit: ${financial.get('npv_mean', 0):,.0f}")
                else:
                    recommendations.append(f"   - Expected Loss: ${financial.get('npv_mean', 0):,.0f}")
                
                # Risk characteristics
                if risk.get('volatility', 0) < 1e6:
                    recommendations.append(f"   + Low Risk Strategy")
                else:
                    recommendations.append(f"   ! High Risk Strategy")
                
                # Operational characteristics
                if operational.get('utilization_mean', 0) > 0.8:
                    recommendations.append(f"   + High Capacity Utilization")
                
                if operational.get('self_sufficiency_mean', 0) > 0.9:
                    recommendations.append(f"   + High Self-Sufficiency Rate")
        
        return recommendations


if __name__ == "__main__":
    # Test code
    print("=== Performance Analyzer Test ===")
    
    # Actual simulation results needed for testing
    # Due to test environment limitations, only basic functionality testing here
    
    analyzer = PerformanceAnalyzer()
    
    # Test basic statistical functions
    test_data = [100, 150, 120, 180, 200, 90, 160, 140, 170, 130]
    
    print(f"Skewness: {analyzer._calculate_skewness(test_data):.3f}")
    print(f"Kurtosis: {analyzer._calculate_kurtosis(test_data):.3f}")
    print(f"Downside Deviation: {analyzer._calculate_downside_deviation(test_data):.3f}")
    print(f"Sortino Ratio: {analyzer._calculate_sortino_ratio(test_data):.3f}")
    print(f"Max Drawdown: {analyzer._calculate_max_drawdown(test_data):.3f}")