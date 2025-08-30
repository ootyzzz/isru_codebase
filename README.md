# ISRU Strategy Simulation System

A comprehensive simulation framework for In-Situ Resource Utilization (ISRU) strategies on the lunar surface, focusing on oxygen production optimization and deployment strategy analysis.

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd isru_codebase

# Install dependencies
pip install -r requirements.txt
```

## Optimal Analysis

Most commonly used commands for optimal analysis:

```bash
# NPV time analysis
python optimal/npv_time_analysis.py

# Generate visualization charts
python optimal/visualization.py

# Run optimization for specific year (config in data/parameters.json)
python optimal/optimal_solu.py
```

## Strategy Simulation

Common command for 3-strategy comparison:

```bash
python strategies/main.py --time-horizon 50 --visualize --n-simulations 100
//n_sim can be set to 1000 e.g. for a more converged demand path.
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

## License

MIT License
