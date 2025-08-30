# ISRU Strategy Simulation System

## Project Overview

This is a simulation system for analyzing and comparing different ISRU (In-Situ Resource Utilization) deployment strategies, focusing on strategy optimization for lunar oxygen production.

## Project Structure

```
isru_codebase/
├── strategies/                    # Strategy simulation module (main functionality)
│   ├── core/                     # Core simulation engine
│   ├── analysis/                 # Analysis tools
│   ├── visualization/            # Visualization module
│   └── utils/                    # Utility modules
├── optimal/                      # Global optimal solution solver
├── models/                       # Mathematical model definitions
├── analysis/                     # Demand analysis module
├── data/                         # Configuration and parameter files
├── examples/                     # Usage examples
├── solvers/                      # Solver interfaces
├── simulation/                   # Simulation tools
└── paper/                        # Paper-related documents
```

## Quick Start

### Environment Setup

#### Method 1: Using Conda Environment (Recommended)

```bash
# Clone the project
git clone <repository-url>
cd isru_codebase

# Create and activate conda environment
conda env create -f environment.yml
conda activate isru

# Verify installation
python strategies/main.py --help
```

#### Method 2: Using pip Installation

```bash
# Clone the project
git clone <repository-url>
cd isru_codebase

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Note: GLPK solver needs to be installed separately
# Windows: Download and install GLPK binary files
# Linux: sudo apt-get install glpk-utils
# Mac: brew install glpk
```

### Basic Usage

```bash
# Run strategy comparison simulation (recommended start)
python strategies/main.py --visualize --time-horizon 20 --n-simulations 100

# Run single strategy simulation
python strategies/main.py single --strategy upfront_deployment --time-horizon 10

# View all available commands
python strategies/main.py --help
```

## Three Strategies

1. **Upfront Deployment**
   - Deploy all capacity in the first year, then maintain operations only
   - Suitable for: Sufficient funding, accurate demand forecasting

2. **Gradual Deployment**
   - Distribute total deployment evenly across all years
   - Suitable for: Limited funding, pursuing steady development

3. **Flexible Deployment**
   - Dynamically adjust based on actual supply-demand differences
   - Suitable for: High market uncertainty, need for flexible response

## Main Features

- **Strategy Comparison Analysis**: Compare performance of different strategies
- **Time Horizon Analysis**: Analyze impact of different time horizons on strategies
- **Monte Carlo Simulation**: Evaluate strategy robustness under uncertainty
- **Visualization Analysis**: Automatically generate charts and analysis reports
- **Global Optimal Solution Comparison**: Compare with theoretical optimal solutions

## Core Dependencies

- **Python 3.13+**
- **numpy**: Numerical computation
- **pandas**: Data processing
- **matplotlib/seaborn**: Visualization
- **pyomo**: Optimization modeling
- **scipy**: Scientific computing (optional)

## Detailed Documentation

- [Strategy Simulation System Details](strategies/README.md)
- [Parameter Configuration Guide](parameter_guide.md)
- [Paper Framework](paper/paper_framework.md)

## Development Guide

### Code Structure

- `strategies/`: Main strategy simulation functionality
- `optimal/`: Global optimal solution solver
- `models/`: Mathematical model definitions
- `data/`: Configuration files and parameters

### Adding New Strategies

1. Define new strategy in `strategies/core/strategy_definitions.py`
2. Implement decision logic in `strategies/core/decision_logic.py`
3. Update related tests and documentation

### Running Tests

```bash
# Run basic functionality tests
python strategies/main.py compare --time-horizon 5 --n-simulations 3

# Run global optimal solution
python optimal/optimal_solu.py
```

## License

MIT

## Contributors

- LGY
- Feifan

## Contact

For questions or suggestions, please contact us through:
- Create an Issue
- Send email to: [add email]

---

**Last Updated**: August 30, 2025
