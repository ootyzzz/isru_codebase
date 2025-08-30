# ISRU Strategy Simulation System - Refactored Version

## System Overview

This is a completely refactored ISRU (In-Situ Resource Utilization) strategy simulation system for analyzing and comparing different deployment strategies in lunar oxygen production.

### Refactoring Highlights

- **True Strategy Simulation**: Rule-driven, not optimization-based
- **Strategy Differentiation**: Three strategies show distinctly different performance characteristics
- **Beautiful Terminal Output**: Progress bars, tables, charts and other visualizations
- **Comprehensive Performance Analysis**: Financial, operational, and risk three-dimensional assessment
- **Batch Simulation Support**: Monte Carlo, time horizon, and parallel analysis

## Project Structure

```
strategies/
├── core/                          # Core simulation engine
│   ├── strategy_definitions.py    # Strategy parameter definitions
│   ├── simulation_engine.py       # Simulation engine
│   ├── decision_logic.py          # Decision logic
│   └── state_manager.py           # State management
├── analysis/                      # Analysis tools
│   ├── batch_runner.py            # Batch simulation executor
│   └── performance_analyzer.py    # Performance analyzer
├── utils/                         # Utility modules
│   └── terminal_display.py        # Terminal display tools
├── visualization/                 # Visualization module
│   ├── strategy_visualizer.py     # Strategy visualizer
│   └── example_usage.py           # Usage examples
├── simulation_results/            # Simulation results storage
│   ├── raw/                       # Raw data (classified by time horizon)
│   ├── summary/                   # Statistical summaries
│   ├── reports/                   # Analysis reports
│   └── exports/                   # Export files
└── main.py                        # Main program entry
```

## Quick Start

### 1. Environment Setup

#### Method 1: Using Conda Environment (Recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate isru

# Ensure in project root directory
cd /path/to/your/isru_codebase
```

#### Method 2: Using pip Installation

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Note: GLPK solver needs separate installation
# Windows: Download and install GLPK binary files
# Linux: sudo apt-get install glpk-utils
# Mac: brew install glpk
```

#### Core Dependencies Description

- **numpy**: Numerical computation foundation library
- **pandas**: Data processing and analysis
- **matplotlib/seaborn**: Data visualization
- **pyomo**: Optimization modeling framework
- **scipy**: Scientific computing library (optional, for advanced statistical analysis)
- **pyyaml**: YAML configuration file parsing

#### Verify Installation

```bash
# Test if environment is correctly configured
python -c "import numpy, pandas, matplotlib, pyomo; print('Environment configured successfully!')"

# Test strategy simulation system
python strategies/main.py --help
```

### 2. Recommended Usage Examples

```bash
# Most common: Run 20-year time horizon strategy comparison simulation with visualization
python strategies/main.py --time-horizon 20 --visualize --n-simulations 100

# Quick test: Run 10-year time horizon strategy comparison simulation (default)
python strategies/main.py --visualize

# Single strategy Monte Carlo simulation
python strategies/main.py monte-carlo --strategy flexible_deployment --time-horizon 30 --n-simulations 500 --visualize

# Strategy comparison analysis and save results
python strategies/main.py compare --time-horizon 25 --n-simulations 200 --visualize --save

# Display visualization charts for existing results
python strategies/main.py visualize

# Export results to Excel
python strategies/main.py results export --strategies upfront_deployment gradual_deployment flexible_deployment --time-horizons 10 20 30
```

### 3. Basic Commands

```bash
# View all available commands
python strategies/main.py --help

# Strategy comparison (recommended start)
python strategies/main.py compare

# Time horizon analysis
python strategies/main.py horizon

# Single simulation
python strategies/main.py single --strategy gradual_deployment
```

## Main Features

### 1. Strategy Comparison Analysis

Compare three strategies under the same conditions:

```bash
# Basic comparison
python strategies/main.py compare --time-horizon 30 --n-simulations 100

# Specified strategy comparison
python strategies/main.py compare --strategies upfront_deployment flexible_deployment --n-simulations 50

# Detailed analysis report
python strategies/main.py compare --detailed-analysis --save
```

**Output Example:**
```
T=10 Year Strategy Comparison
┌────────────────────┬───────────────┬───────────────┬───────────────┐
│ Metric             │ Upfront_Depl  │ Gradual_Depl  │ Flexible_Depl │
├────────────────────┼───────────────┼───────────────┼───────────────┤
│ NPV Mean           │          1.3M │          1.7M │          1.7M │
│ NPV Std Dev        │        327.1K │        524.9K │        509.6K │
│ Avg Utilization    │         87.4% │         53.5% │         65.8% │
│ Self-Sufficiency   │         72.4% │         84.2% │         83.3% │
└────────────────────┴───────────────┴───────────────┴───────────────┘
```

### 2. Time Horizon Impact Analysis

Analyze the impact of different time horizons on strategy performance:

```bash
# Standard time horizon analysis
python strategies/main.py horizon --time-horizons 10 20 30 40 50

# Custom time horizons
python strategies/main.py horizon --time-horizons 15 25 35 --strategies gradual_deployment flexible_deployment
```

### 3. Monte Carlo Simulation

Perform large-scale random simulations for a single strategy:

```bash
# Monte Carlo simulation
python strategies/main.py monte-carlo --strategy upfront_deployment --n-simulations 1000

# Save results
python strategies/main.py monte-carlo --strategy flexible_deployment --n-simulations 500 --save
```

### 4. Parallel Batch Simulation

Efficiently execute large-scale simulations:

```bash
# Parallel simulation of all strategies and time horizons
python strategies/main.py parallel --n-simulations 100

# Specify number of parallel processes
python strategies/main.py parallel --max-workers 4 --time-horizons 10 20 30
```

## Three Strategies Detailed

### 1. Upfront Deployment Strategy
- **Characteristics**: Large upfront investment, deploy all capacity at once
- **Deployment Method**: Deploy total final deployment in the first year (based on T-year demand forecast)
- **Subsequent Years**: No new deployments, only maintain operations
- **Advantages**: Avoid later expansion costs, fully utilize economies of scale
- **Disadvantages**: High upfront capital pressure, potential overcapacity
- **Applicable Scenarios**: Sufficient funding, accurate demand forecasting, pursuing long-term stability

### 2. Gradual Deployment Strategy
- **Characteristics**: Evenly distributed investment, steady expansion
- **Deployment Method**: Distribute total deployment evenly across T years
- **Calculation Method**: Annual increment = Final total deployment / T
- **Assumption**: System knows time horizon T at the beginning, can determine annual increments
- **Advantages**: Distributed capital pressure, relatively low risk
- **Disadvantages**: May not meet early demand, requires Earth supply
- **Applicable Scenarios**: Limited funding, pursuing steady development, risk averse

### 3. Flexible Deployment Strategy
- **Characteristics**: Responsive deployment, dynamically adjust based on actual supply-demand conditions
- **Deployment Logic**: If year t supply is n less than demand, then deploy additional n in year t+1
- **Decision Rules**:
  - If Supply_t < Demand_t, then New_Deployment_{t+1} = Demand_t - Supply_t
  - If Supply_t ≥ Demand_t, then New_Deployment_{t+1} = 0
- **Characteristics**: Reactive strategy, always trying to fill the previous year's gap
- **Advantages**: Respond to market demand, avoid over-investment
- **Disadvantages**: May have lag, cannot plan predictively
- **Applicable Scenarios**: High market uncertainty, need flexible response

## Performance Metrics Description

### Financial Metrics
- **NPV (Net Present Value)**: Financial value of the project
- **NPV Standard Deviation**: Revenue volatility
- **Success Rate**: Probability of positive NPV
- **Sharpe Ratio**: Risk-adjusted return

### Operational Metrics
- **Average Utilization**: Capacity usage efficiency
- **Self-Sufficiency Rate**: Local production as proportion of total demand
- **Capacity Expansion Count**: Strategy expansion frequency
- **Final Capacity**: Total capacity at project end

### Risk Metrics
- **Volatility**: Standard deviation of NPV
- **Downside Risk**: Risk of negative returns
- **Maximum Drawdown**: Maximum loss magnitude
- **Sortino Ratio**: Downside risk-adjusted return

## Advanced Usage

### 1. Custom Parameters

Modify parameters in `data/parameters.json`:

```json
{
  "demand": {
    "D0": 10,        // Initial demand
    "mu": 0.2,       // Demand growth rate
    "sigma": 0.2     // Demand volatility
  },
  "costs": {
    "c_dev": 10000,  // Development cost
    "c_op": 3000,    // Operating cost
    "c_E": 20000     // Earth supply cost
  }
}
```

### 2. Result Files

Simulation results are automatically saved in `strategies/simulation_results/` directory:

```
simulation_results/
├── T10/
│   ├── upfront_deployment_detailed.json    # Detailed simulation data
│   ├── upfront_deployment_summary.json     # Statistical summary
│   └── ...
├── T20/
└── ...
```

### 3. Global Optimal Solution Comparison

```bash
# Compare with optimal solution from test_fixed_model.py
python strategies/main.py optimal --time-horizon 30
```

## Visualization Features

The system provides two visualization methods:

### Terminal Visualization
- **Progress Bars**: Real-time simulation progress display
- **Tables**: Structured comparison results display
- **Summary Boxes**: Highlight key metrics
- **ASCII Charts**: Simple trend visualization

### Graphical Visualization
Use `--visualize` parameter to enable graphical visualization:

```bash
# Enable visualization charts
python strategies/main.py --visualize --time-horizon 50 --n-simulations 100
```

**Visualization Charts Include:**
1. **Decision Variable Comparison Charts**: Production, capacity, inventory, Earth supply, capacity expansion, utilization
2. **Demand vs Supply Comparison Charts**: Demand curves vs strategy supply capacity comparison
3. **Cost Analysis Charts**: Cost composition stacked charts and NPV comparison charts

**Chart Features:**
- Support English display
- Three strategies distinguished by different colors
- Interactive charts, zoomable
- Auto-adapt to time horizons (T10, T20, T30, T40, T50)

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'seaborn'**
   - Solution: Visualization module is optional, does not affect core functionality

2. **Strategy results are identical**
   - Check: Ensure using new simulation system, not old strategy_runner.py

3. **Out of memory**
   - Solution: Reduce simulation count or use parallel simulation

### Performance Optimization

```bash
# Reduce simulation count for quick testing
python strategies/main.py compare --n-simulations 20

# Use parallel processing for efficiency
python strategies/main.py parallel --max-workers 8
```

## Technical Architecture

### Core Design Principles

1. **Rule-Driven**: Strategies execute based on predefined rules, not optimization solving
2. **State Management**: Clear state transitions and historical records
3. **Modular**: Core components are independent, easy to extend
4. **Visualization**: Beautiful terminal output and progress display

### Extension Guide

Adding new strategies:

```python
# Add in strategies/core/strategy_definitions.py
@staticmethod
def get_new_strategy() -> StrategyParams:
    return StrategyParams(
        name="new_strategy",
        description="New strategy description",
        initial_deployment_ratio=0.8,
        utilization_threshold=0.75,
        expansion_ratio=0.3,
        risk_tolerance=0.6,
        cost_sensitivity=0.4
    )
```

## Usage Recommendations

### Research Scenarios

1. **Strategy Selection**: Use `compare` command to compare strategies
2. **Time Planning**: Use `horizon` to analyze long-term impacts
3. **Risk Assessment**: Focus on NPV standard deviation and success rate
4. **Sensitivity Analysis**: Re-simulate after modifying parameters

### Best Practices

1. **Start Small**: Begin testing with small-scale simulations
2. **Save Results**: Use `--save` parameter for important analyses
3. **Multiple Runs**: Use different random seeds to verify results
4. **Document**: Record parameter settings and analysis conclusions

## Support

For questions or suggestions, please check:

1. Whether parameter file format is correct
2. Whether environment dependencies are satisfied
3. Whether file paths are correct
4. Whether simulation parameters are reasonable

---

**Refactoring Completion Date**: August 27, 2025  
**Version**: 2.0  
**Status**: Production Ready