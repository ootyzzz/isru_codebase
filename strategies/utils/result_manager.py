#!/usr/bin/env python3
"""
Result Manager
Provide comprehensive simulation result saving, loading and management functionality
"""

import json
import csv
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from strategies.core.simulation_engine import SimulationResult


class ResultManager:
    """Result manager"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize result manager
        
        Args:
            base_dir: Base save directory
        """
        self.base_dir = base_dir or Path(__file__).parent.parent / "simulation_results"
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectory structure
        self.raw_dir = self.base_dir / "raw"           # Raw simulation data
        self.summary_dir = self.base_dir / "summary"   # Statistical summaries
        self.reports_dir = self.base_dir / "reports"   # Analysis reports
        self.exports_dir = self.base_dir / "exports"   # Export files
        
        for dir_path in [self.raw_dir, self.summary_dir, self.reports_dir, self.exports_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_simulation_batch(self,
                            strategy_name: str,
                            T: int,
                            results: List[SimulationResult],
                            stats: Dict[str, float],
                            metadata: Optional[Dict] = None) -> Dict[str, Path]:
        """
        Save batch simulation results
        
        Args:
            strategy_name: Strategy name
            T: Time horizon
            results: Simulation results list
            stats: Statistical information
            metadata: Metadata information
            
        Returns:
            Dictionary of saved file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create time horizon directory
        time_dir = self.raw_dir / f"T{T}"
        time_dir.mkdir(exist_ok=True)
        
        summary_time_dir = self.summary_dir / f"T{T}"
        summary_time_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # 1. Save raw detailed data
        detailed_file = time_dir / f"{strategy_name}_detailed_{timestamp}.json"
        detailed_data = {
            'metadata': {
                'strategy': strategy_name,
                'time_horizon': T,
                'n_simulations': len(results),
                'timestamp': timestamp,
                'version': '2.0',
                **(metadata or {})
            },
            'results': [self._convert_to_serializable(result.to_dict()) for result in results]
        }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        saved_files['detailed'] = detailed_file
        
        # 2. Save statistical summary
        summary_file = summary_time_dir / f"{strategy_name}_summary_{timestamp}.json"
        summary_data = {
            'metadata': detailed_data['metadata'],
            'statistics': self._convert_to_serializable(stats),
            'performance_metrics': self._convert_to_serializable(self._calculate_extended_metrics(results))
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        saved_files['summary'] = summary_file
        
        # 3. Save latest version (overwrite)
        latest_detailed = time_dir / f"{strategy_name}_latest.json"
        latest_summary = summary_time_dir / f"{strategy_name}_latest.json"
        
        with open(latest_detailed, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False)
        
        with open(latest_summary, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        saved_files['latest_detailed'] = latest_detailed
        saved_files['latest_summary'] = latest_summary
        
        # 4. Save as CSV format (for Excel analysis)
        csv_file = self.exports_dir / f"{strategy_name}_T{T}_{timestamp}.csv"
        self._save_as_csv(results, stats, csv_file)
        saved_files['csv'] = csv_file
        
        # 5. Save as pickle format (Python objects)
        pickle_file = time_dir / f"{strategy_name}_objects_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'stats': stats,
                'metadata': metadata
            }, f)
        saved_files['pickle'] = pickle_file
        
        return saved_files
    
    def save_comparison_results(self,
                              comparison_data: Dict[str, Any],
                              T: int,
                              analysis_type: str = "strategy_comparison") -> Path:
        """
        Save strategy comparison results
        
        Args:
            comparison_data: Comparison data
            T: Time horizon
            analysis_type: Analysis type
            
        Returns:
            Saved file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to reports directory
        report_file = self.reports_dir / f"{analysis_type}_T{T}_{timestamp}.json"
        
        report_data = {
            'metadata': {
                'analysis_type': analysis_type,
                'time_horizon': T,
                'timestamp': timestamp,
                'strategies': list(comparison_data.keys()) if isinstance(comparison_data, dict) else [],
                'version': '2.0'
            },
            'comparison_data': self._convert_to_serializable(comparison_data),
            'summary': self._convert_to_serializable(self._create_comparison_summary(comparison_data))
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Also save latest version
        latest_file = self.reports_dir / f"{analysis_type}_T{T}_latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return report_file
    
    def save_horizon_analysis(self,
                            horizon_results: Dict[int, Dict[str, Any]],
                            strategies: List[str]) -> Path:
        """
        Save time horizon analysis results
        
        Args:
            horizon_results: Time horizon analysis results
            strategies: Strategy list
            
        Returns:
            Saved file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_file = self.reports_dir / f"horizon_analysis_{timestamp}.json"
        
        report_data = {
            'metadata': {
                'analysis_type': 'horizon_analysis',
                'time_horizons': list(horizon_results.keys()),
                'strategies': strategies,
                'timestamp': timestamp,
                'version': '2.0'
            },
            'horizon_results': self._convert_to_serializable(horizon_results),
            'trend_analysis': self._convert_to_serializable(self._analyze_horizon_trends(horizon_results, strategies))
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Save latest version
        latest_file = self.reports_dir / "horizon_analysis_latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return report_file
    
    def load_simulation_results(self,
                              strategy_name: str,
                              T: int,
                              version: str = "latest") -> Optional[Dict]:
        """
        Load simulation results
        
        Args:
            strategy_name: Strategy name
            T: Time horizon
            version: Version ("latest" or timestamp)
            
        Returns:
            Simulation result data
        """
        if version == "latest":
            file_path = self.raw_dir / f"T{T}" / f"{strategy_name}_latest.json"
        else:
            file_path = self.raw_dir / f"T{T}" / f"{strategy_name}_detailed_{version}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_summary_stats(self,
                         strategy_name: str,
                         T: int,
                         version: str = "latest") -> Optional[Dict]:
        """
        Load statistical summary
        
        Args:
            strategy_name: Strategy name
            T: Time horizon
            version: Version ("latest" or timestamp)
            
        Returns:
            Statistical summary data
        """
        if version == "latest":
            file_path = self.summary_dir / f"T{T}" / f"{strategy_name}_latest.json"
        else:
            file_path = self.summary_dir / f"T{T}" / f"{strategy_name}_summary_{version}.json"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def export_to_excel(self,
                       strategies: List[str],
                       time_horizons: List[int],
                       output_file: Optional[Path] = None) -> Path:
        """
        Export results to Excel file
        
        Args:
            strategies: Strategy list
            time_horizons: Time horizon list
            output_file: Output file path
            
        Returns:
            Excel file path
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.exports_dir / f"simulation_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Create summary table
            summary_data = []
            
            for T in time_horizons:
                for strategy in strategies:
                    stats = self.load_summary_stats(strategy, T)
                    if stats:
                        summary_data.append({
                            'Time Horizon': T,
                            'Strategy': strategy,
                            'NPV Mean': stats['statistics']['npv_mean'],
                            'NPV Std Dev': stats['statistics']['npv_std'],
                            'Avg Utilization': stats['statistics']['utilization_mean'],
                            'Self-Sufficiency': stats['statistics']['self_sufficiency_mean'],
                            'Success Rate': stats['statistics']['probability_positive_npv']
                        })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create detailed table for each strategy
            for strategy in strategies:
                strategy_data = []
                
                for T in time_horizons:
                    results_data = self.load_simulation_results(strategy, T)
                    if results_data:
                        for i, result in enumerate(results_data['results']):
                            strategy_data.append({
                                'Time Horizon': T,
                                'Simulation No': i + 1,
                                'NPV': result['performance_metrics']['npv'],
                                'Total Demand': result['performance_metrics']['total_demand'],
                                'Total Production': result['performance_metrics']['total_production'],
                                'Earth Supply': result['performance_metrics']['total_earth_supply'],
                                'Avg Utilization': result['performance_metrics']['avg_utilization'],
                                'Self-Sufficiency': result['performance_metrics']['self_sufficiency_rate']
                            })
                
                if strategy_data:
                    strategy_df = pd.DataFrame(strategy_data)
                    strategy_df.to_excel(writer, sheet_name=strategy.title(), index=False)
        
        return output_file
    
    def get_available_results(self) -> Dict[str, List[str]]:
        """
        Get available results list
        
        Returns:
            Dictionary of available results
        """
        available = {}
        
        for time_dir in self.raw_dir.glob("T*"):
            T = time_dir.name
            available[T] = []
            
            for file_path in time_dir.glob("*_latest.json"):
                strategy_name = file_path.stem.replace("_latest", "")
                available[T].append(strategy_name)
        
        return available
    
    def cleanup_old_results(self, keep_days: int = 30):
        """
        Clean up old result files
        
        Args:
            keep_days: Days to keep
        """
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
        
        for dir_path in [self.raw_dir, self.summary_dir, self.reports_dir]:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith("_latest.json"):
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        print(f"Deleted old file: {file_path}")
    
    def _save_as_csv(self,
                    results: List[SimulationResult],
                    stats: Dict[str, float],
                    csv_file: Path):
        """Save as CSV format"""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Simulation No', 'NPV', 'Total Demand', 'Total Production', 'Earth Supply',
                'Avg Utilization', 'Self-Sufficiency', 'Final Capacity', 'Capacity Expansions'
            ])
            
            # Write data
            for i, result in enumerate(results):
                metrics = result.performance_metrics
                writer.writerow([
                    i + 1,
                    metrics.get('npv', 0),
                    metrics.get('total_demand', 0),
                    metrics.get('total_production', 0),
                    metrics.get('total_earth_supply', 0),
                    metrics.get('avg_utilization', 0),
                    metrics.get('self_sufficiency_rate', 0),
                    metrics.get('final_capacity', 0),
                    metrics.get('capacity_expansions', 0)
                ])
    
    def _calculate_extended_metrics(self, results: List[SimulationResult]) -> Dict[str, float]:
        """Calculate extended performance metrics"""
        if not results:
            return {}
        
        npvs = [r.performance_metrics.get('npv', 0) for r in results]
        
        return {
            'npv_median': float(np.median(npvs)),
            'npv_q25': float(np.percentile(npvs, 25)),
            'npv_q75': float(np.percentile(npvs, 75)),
            'npv_iqr': float(np.percentile(npvs, 75) - np.percentile(npvs, 25)),
            'npv_skewness': float(self._calculate_skewness(npvs)),
            'success_rate_50pct': float(np.mean([npv > np.median(npvs) for npv in npvs])),
            'worst_case_5pct': float(np.percentile(npvs, 5)),
            'best_case_95pct': float(np.percentile(npvs, 95))
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return float(np.mean([((x - mean) / std) ** 3 for x in data]))
    
    def _create_comparison_summary(self, comparison_data: Dict) -> Dict:
        """Create comparison summary"""
        if not comparison_data:
            return {}
        
        # Extract NPV data for ranking
        npv_data = {}
        for strategy, (results, stats) in comparison_data.items():
            npv_data[strategy] = stats.get('npv_mean', 0)
        
        # Sort
        sorted_strategies = sorted(npv_data.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'best_strategy': sorted_strategies[0][0] if sorted_strategies else None,
            'worst_strategy': sorted_strategies[-1][0] if sorted_strategies else None,
            'npv_ranking': [{'strategy': s, 'npv': v} for s, v in sorted_strategies],
            'performance_gap': sorted_strategies[0][1] - sorted_strategies[-1][1] if len(sorted_strategies) >= 2 else 0
        }
    
    def _analyze_horizon_trends(self,
                              horizon_results: Dict[int, Dict],
                              strategies: List[str]) -> Dict:
        """Analyze time horizon trends"""
        trends = {}
        
        for strategy in strategies:
            strategy_data = {}
            horizons = sorted(horizon_results.keys())
            
            npv_values = []
            for T in horizons:
                if strategy in horizon_results[T]:
                    _, stats = horizon_results[T][strategy]
                    npv_values.append(stats.get('npv_mean', 0))
                else:
                    npv_values.append(0)
            
            if len(npv_values) >= 2:
                # Calculate trend slope
                slope = np.polyfit(horizons, npv_values, 1)[0]
                
                strategy_data = {
                    'npv_trend_slope': float(slope),
                    'is_improving': slope > 0,
                    'growth_rate': float(slope / npv_values[0]) if npv_values[0] > 0 else 0,
                    'npv_values': npv_values,
                    'time_horizons': horizons
                }
            
            trends[strategy] = strategy_data
        
        return trends
    
    def _convert_to_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'to_dict'):  # Handle SimulationResult and other objects
            return self._convert_to_serializable(obj.to_dict())
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        else:
            return obj


if __name__ == "__main__":
    # Test code
    print("=== Result Manager Test ===")
    
    manager = ResultManager()
    
    # Test directory creation
    print(f"Base directory: {manager.base_dir}")
    print(f"Raw data directory: {manager.raw_dir}")
    print(f"Summary directory: {manager.summary_dir}")
    print(f"Reports directory: {manager.reports_dir}")
    print(f"Export directory: {manager.exports_dir}")
    
    # Test available results query
    available = manager.get_available_results()
    print(f"Available results: {available}")