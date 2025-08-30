#!/usr/bin/env python3
"""
ISRU Strategy Visualizer
Implements multi-strategy decision variable comparison visualization and analysis chart generation

Main features:
    - Decision variables comparison charts: production, capacity, inventory, earth supply, capacity expansion, utilization
    - Demand vs supply comparison charts: demand curves vs strategy supply capacity comparison
    - Cost analysis charts: cost composition stacked charts and NPV comparison charts

Usage examples:
    # Basic usage
    plotter = DecisionVariablesPlotter()
    figures = plotter.create_comprehensive_dashboard("strategies/simulation_results", time_horizon=50)
    
    # Generate individual charts
    strategies_data = plotter.load_simulation_data("strategies/simulation_results", 30)
    fig1 = plotter.plot_decision_variables(strategies_data)
    fig2 = plotter.plot_demand_vs_supply(strategies_data)
    fig3 = plotter.plot_cost_analysis(strategies_data)

Supported time horizons: T10, T20, T30, T40, T50
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Set matplotlib to interactive mode
plt.ion()

# Ignore some matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class DecisionVariablesPlotter:
    """
    ISRU Strategy Visualization Plotter
    
    Used to generate visualization charts for ISRU strategy simulation results, supporting multi-strategy comparison analysis.
    Automatically adapts to data from different time horizons, providing interactive charts.
    
    Attributes:
        figsize: Chart size (width, height)
        strategy_colors: Strategy color mapping
        strategy_labels: Strategy label mapping
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Initialize plotter
        
        Args:
            figsize: Chart size (width, height)
        """
        self.figsize = figsize
        self.strategy_colors = {
            'upfront_deployment': '#2E8B57',    # Sea green - Upfront deployment
            'gradual_deployment': '#4169E1',    # Royal blue - Gradual deployment
            'flexible_deployment': '#DC143C'    # Deep red - Flexible deployment
        }
        self.strategy_labels = {
            'upfront_deployment': 'Upfront Deployment',
            'gradual_deployment': 'Gradual Deployment',
            'flexible_deployment': 'Flexible Deployment'
        }
        
        # Set fonts
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_simulation_data(self, results_dir: str = "strategies/simulation_results", time_horizon: int = 10) -> Dict[str, Any]:
        """
        Load simulation result data
        
        Args:
            results_dir: Results directory path
            time_horizon: Time horizon (years)
            
        Returns:
            Dictionary containing data for three strategies
        """
        results_path = Path(results_dir)
        strategies_data = {}
        
        # Load latest results for three strategies
        for strategy in ['upfront_deployment', 'gradual_deployment', 'flexible_deployment']:
            latest_file = results_path / "raw" / f"T{time_horizon}" / f"{strategy}_latest.json"
            
            if latest_file.exists():
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    strategies_data[strategy] = data
            else:
                print(f"Warning: Data file not found for {strategy} strategy: {latest_file}")
                
        return strategies_data
    
    def extract_decision_variables(self, simulation_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Extract decision variables from simulation data
        
        Args:
            simulation_data: Simulation data for a single strategy
            
        Returns:
            Decision variables dictionary
        """
        if not simulation_data or 'results' not in simulation_data:
            return {}
            
        # Take first simulation result (usually contains results from multiple random seeds)
        result = simulation_data['results'][0]
        
        # Extract time series
        time_steps = list(range(len(result['decisions'])))
        
        # Extract various decision variables
        variables = {
            'time_steps': time_steps,
            'production': [],           # Production
            'capacity': [],            # Capacity
            'inventory': [],           # Inventory
            'earth_supply': [],        # Earth supply
            'capacity_expansion': [],  # Capacity expansion
            'utilization': [],         # Utilization
            'demand': result.get('demand_path', [])  # Demand path
        }
        
        # Extract data from decisions
        for decision in result['decisions']:
            variables['production'].append(decision.get('planned_production', 0))
            variables['capacity_expansion'].append(decision.get('capacity_expansion', 0))
            variables['earth_supply'].append(decision.get('earth_supply_request', 0))
        
        # Extract data from states
        for state in result['states']:
            variables['capacity'].append(state.get('total_capacity', 0))
            variables['inventory'].append(state.get('inventory', 0))
            
            # Calculate utilization
            capacity = state.get('total_capacity', 1)
            production = state.get('actual_production', 0)
            utilization = production / capacity if capacity > 0 else 0
            variables['utilization'].append(utilization)
        
        return variables
    
    def plot_decision_variables(self, strategies_data: Dict[str, Any], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot decision variables comparison chart
        
        Args:
            strategies_data: Dictionary containing all strategy data
            save_path: Save path (optional)
            
        Returns:
            matplotlib figure object
        """
        # Create subplot layout (3 rows, 2 columns)
        fig, axes = plt.subplots(3, 2, figsize=self.figsize)
        fig.suptitle('ISRU Decision Variables Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Flatten axes array for indexing
        axes_flat = axes.flatten()
        
        # Define variables to plot and corresponding subplots
        plot_configs = [
            ('production', 'Production (kg)', 0),
            ('capacity', 'Capacity (kg)', 1),
            ('inventory', 'Inventory Level (kg)', 2),
            ('earth_supply', 'Earth Supply (kg)', 3),
            ('capacity_expansion', 'Capacity Expansion (kg)', 4),
            ('utilization', 'Utilization Rate', 5)
        ]
        
        # Extract data for each strategy and plot
        all_variables = {}
        for strategy_name, data in strategies_data.items():
            if data:
                all_variables[strategy_name] = self.extract_decision_variables(data)
        
        # Plot each variable
        for var_name, var_label, ax_idx in plot_configs:
            ax = axes_flat[ax_idx]
            
            for strategy_name, variables in all_variables.items():
                if var_name in variables and variables[var_name]:
                    time_steps = variables.get('time_steps', range(len(variables[var_name])))
                    values = variables[var_name]
                    
                    # Ensure time steps and values have consistent length
                    min_len = min(len(time_steps), len(values))
                    time_steps_plot = time_steps[:min_len]
                    values_plot = values[:min_len]
                    
                    ax.plot(time_steps_plot, values_plot, 
                           color=self.strategy_colors[strategy_name],
                           label=self.strategy_labels[strategy_name],
                           linewidth=2, marker='o', markersize=4)
            
            ax.set_title(var_label, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step (Years)', fontsize=10)
            ax.set_ylabel(var_label, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Special handling for utilization chart
            if var_name == 'utilization':
                ax.set_ylim(0, 1.1)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Full Capacity')
        
        # Adjust layout
        plt.tight_layout()
        
        # Show chart (interactive mode)
        plt.show()
        
        # Save chart (if path specified)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        return fig
    
    def plot_demand_vs_supply(self, strategies_data: Dict[str, Any]) -> plt.Figure:
        """
        Plot demand vs supply comparison chart - Fixed version
        
        Args:
            strategies_data: Dictionary containing all strategy data
            
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        demand_plotted = False  # Flag whether demand line has been plotted
        
        for strategy_name, data in strategies_data.items():
            if not data:
                continue
                
            variables = self.extract_decision_variables(data)
            if not variables:
                continue
                
            time_steps = variables.get('time_steps', [])
            demand = variables.get('demand', [])
            production = variables.get('production', [])
            earth_supply = variables.get('earth_supply', [])
            
            # Calculate total supply (production + earth supply)
            total_supply = [p + e for p, e in zip(production, earth_supply)]
            
            # Plot demand line (demand should be same for all strategies, plot only once)
            if not demand_plotted and demand:
                # Demand data usually has one more point than time steps (includes initial value), create corresponding time axis
                demand_time_steps = list(range(len(demand)))
                ax.plot(demand_time_steps, demand,
                       color='black', linewidth=3, linestyle='--',
                       label='Demand', alpha=0.8)
                demand_plotted = True
            
            # Plot total supply line
            min_len = min(len(time_steps), len(total_supply))
            if min_len > 0:
                ax.plot(time_steps[:min_len], total_supply[:min_len],
                       color=self.strategy_colors[strategy_name],
                       label=f'{self.strategy_labels[strategy_name]} - Total Supply',
                       linewidth=2, marker='s', markersize=4)
        
        ax.set_title('Demand vs Supply Comparison Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step (Years)', fontsize=12)
        ax.set_ylabel('Quantity (kg)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_cost_analysis(self, strategies_data: Dict[str, Any]) -> plt.Figure:
        """
        Plot cost analysis chart
        
        Args:
            strategies_data: Dictionary containing all strategy data
            
        Returns:
            matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract cost data
        strategies = []
        expansion_costs = []
        operational_costs = []
        supply_costs = []
        total_costs = []
        npvs = []
        
        for strategy_name, data in strategies_data.items():
            if not data or 'results' not in data:
                continue
                
            result = data['results'][0]
            metrics = result.get('performance_metrics', {})
            
            strategies.append(self.strategy_labels[strategy_name])
            expansion_costs.append(metrics.get('total_expansion_cost', 0))
            operational_costs.append(metrics.get('total_operational_cost', 0))
            supply_costs.append(metrics.get('total_supply_cost', 0))
            total_costs.append(metrics.get('total_cost', 0))
            npvs.append(metrics.get('npv', 0))
        
        # Plot cost composition stacked bar chart
        x = np.arange(len(strategies))
        width = 0.6
        
        ax1.bar(x, expansion_costs, width, label='Expansion Cost',
               color='#FF6B6B', alpha=0.8)
        ax1.bar(x, operational_costs, width, bottom=expansion_costs,
               label='Operational Cost', color='#4ECDC4', alpha=0.8)
        ax1.bar(x, supply_costs, width,
               bottom=[e+o for e,o in zip(expansion_costs, operational_costs)],
               label='Supply Cost', color='#45B7D1', alpha=0.8)
        
        ax1.set_title('Cost Composition Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost (10K CNY)', fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot NPV comparison
        colors = [self.strategy_colors[name] for name in strategies_data.keys() if strategies_data[name]]
        bars = ax2.bar(x, npvs, width, color=colors, alpha=0.8)
        
        ax2.set_title('Net Present Value (NPV) Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylabel('NPV (10K CNY)', fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategies)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bar chart
        for bar, npv in zip(bars, npvs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{npv/10000:.1f}0K',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_npv_cdf(self, strategies_data: Dict[str, Any]) -> plt.Figure:
        """
        Plot NPV Cumulative Distribution Function (CDF) chart
        
        Args:
            strategies_data: Dictionary containing all strategy data
            
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for strategy_name, data in strategies_data.items():
            if not data or 'results' not in data:
                continue
                
            # Extract NPV values from all simulations
            npv_values = []
            for result in data['results']:
                npv = result.get('performance_metrics', {}).get('npv', 0)
                npv_values.append(npv / 10000)  # Convert to 10K CNY
            
            if npv_values:
                # Sort NPV values
                npv_sorted = np.sort(npv_values)
                # Calculate cumulative probability
                cumulative_prob = np.arange(1, len(npv_sorted) + 1) / len(npv_sorted)
                
                # Plot CDF curve
                ax.plot(npv_sorted, cumulative_prob,
                       color=self.strategy_colors[strategy_name],
                       label=self.strategy_labels[strategy_name],
                       linewidth=2, marker='o', markersize=3, alpha=0.8)
                
                # Add statistical information
                mean_npv = np.mean(npv_values)
                median_npv = np.median(npv_values)
                
                # Mark mean and median on chart
                mean_prob = np.interp(mean_npv, npv_sorted, cumulative_prob)
                median_prob = 0.5
                
                ax.axvline(x=mean_npv, color=self.strategy_colors[strategy_name],
                          linestyle='--', alpha=0.8, linewidth=2,
                          label=f'E[NPV] {self.strategy_labels[strategy_name]}')
                ax.axvline(x=median_npv, color=self.strategy_colors[strategy_name],
                          linestyle=':', alpha=0.6, linewidth=1)
        
        # Add zero NPV reference line
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.5, linewidth=1, label='Break-even (NPV=0)')
        
        ax.set_title('NPV Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
        ax.set_xlabel('NPV (10K CNY)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Set y-axis range to 0-1
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def create_comprehensive_dashboard(self, results_dir: str = "strategies/simulation_results", time_horizon: int = 10) -> List[plt.Figure]:
        """
        Create comprehensive dashboard
        
        Args:
            results_dir: Results directory path
            time_horizon: Time horizon (years)
            
        Returns:
            List of figure objects
        """
        print("Loading simulation data...")
        strategies_data = self.load_simulation_data(results_dir, time_horizon)
        
        if not strategies_data:
            print("Error: No strategy data found")
            return []
        
        print(f"Loaded data for {len(strategies_data)} strategies")
        
        figures = []
        
        try:
            # 1. Main decision variables comparison chart
            print("Generating decision variables comparison chart...")
            fig1 = self.plot_decision_variables(strategies_data)
            figures.append(fig1)
            
            # 2. Demand vs supply comparison chart
            print("Generating demand vs supply comparison chart...")
            fig2 = self.plot_demand_vs_supply(strategies_data)
            figures.append(fig2)
            
            # 3. Cost analysis chart
            print("Generating cost analysis chart...")
            fig3 = self.plot_cost_analysis(strategies_data)
            figures.append(fig3)
            
            # 4. NPV cumulative distribution function chart
            print("Generating NPV CDF chart...")
            fig4 = self.plot_npv_cdf(strategies_data)
            figures.append(fig4)
            
            print("All charts generated successfully!")
        except Exception as e:
            print(f"Error occurred while generating charts: {e}")
            
        return figures


def main():
    """
    Main function - for testing and standalone execution
    
    Usage example:
        python strategies/visualization/strategy_visualizer.py
    """
    print("=== ISRU Strategy Visualizer Test ===")
    
    # Create plotter
    plotter = DecisionVariablesPlotter(figsize=(16, 12))
    
    # Test different time horizons
    for time_horizon in [10, 50]:
        print(f"\nTesting time horizon: T{time_horizon}")
        
        # Generate comprehensive dashboard
        figures = plotter.create_comprehensive_dashboard(
            results_dir="strategies/simulation_results",
            time_horizon=time_horizon
        )
        
        if figures:
            print(f"Successfully generated {len(figures)} charts")
            
            # Keep charts displayed
            input(f"Press Enter to close T{time_horizon} charts...")
            plt.close('all')
        else:
            print(f"Failed to generate T{time_horizon} charts")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()