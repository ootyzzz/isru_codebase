# ISRU Oxygen Production Optimization Model Formulation

## Overview

This document presents the mathematical formulation of the In-Situ Resource Utilization (ISRU) oxygen production optimization model for lunar missions. The model optimizes the deployment and operation of oxygen production systems under demand uncertainty.

## Model Components

### 1. Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| Qt[t] | R+ | Delivered oxygen quantity at time t [kg] |
| Qt_cap[t] | R+ | Production capacity at time t [kg] |
| St[t] | R+ | Shortage quantity at time t [kg] |
| Et[t] | R+ | Excess quantity at time t [kg] |
| Mt[t] | R+ | Deployed ISRU mass at time t [kg] |
| delta_Mt[t] | R+ | New ISRU deployment at time t [kg] |
| M_leo[t] | R+ | Mass launched to LEO at time t [kg] |
| Q_earth[t] | R+ | Earth-supplied oxygen quantity at time t [kg] |

### 2. Parameters

#### Economic Parameters
- T: Planning horizon [years]
- r: Discount rate
- P_m: Market price of oxygen [$/kg]

#### Technology Parameters
- eta: ISRU production efficiency [kg oxygen/kg ISRU/year]
- alpha: Launch mass factor (total mass/ISRU mass)
- beta: Byproduct factor
- M0: Initial ISRU mass [kg]

#### Cost Parameters
- c_L: Launch cost [$/kg]
- c_dev: Development cost [$/kg ISRU]
- c_op: Operating cost [$/kg ISRU/year]
- c_bu: Shortage penalty cost [$/kg]
- c_S: Storage cost [$/kg]
- c_by: Byproduct cost [$/kg]
- c_E: Earth supply cost [$/kg]

#### Demand Parameters
- D0: Initial demand [kg/year]
- mu: Demand growth rate
- sigma: Demand volatility

### 3. Objective Function

Maximize Net Present Value (NPV):

```
NPV = Revenue - Total_Cost
```

Where:

**Revenue:**
```
Revenue = Σ(t=0 to T) P_m * Qt[t] * (1+r)^(-t)
```

**Total Cost:**
```
Total_Cost = Launch_Cost + Development_Cost + Operating_Cost + 
             Shortage_Cost + Storage_Cost + Byproduct_Cost + Earth_Supply_Cost
```

**Cost Components:**
- Launch Cost: `Σ(t=0 to T) c_L * M_leo[t] * (1+r)^(-t)`
- Development Cost: `Σ(t=0 to T) c_dev * delta_Mt[t] * (1+r)^(-t)`
- Operating Cost: `Σ(t=0 to T) c_op * Mt[t] * (1+r)^(-t)`
- Shortage Cost: `Σ(t=0 to T) c_bu * St[t] * (1+r)^(-t)`
- Storage Cost: `Σ(t=0 to T) c_S * Et[t] * (1+r)^(-t)`
- Byproduct Cost: `Σ(t=0 to T) c_by * beta * Qt[t] * (1+r)^(-t)`
- Earth Supply Cost: `Σ(t=0 to T) c_E * Q_earth[t] * (1+r)^(-t)`

### 4. Constraints

#### 4.1 Demand Balance Constraint
```
Qt[t] + Q_earth[t] + St[t] = Demand[t]  ∀t ∈ T
```
Ensures that total supply (ISRU delivery + Earth supply + shortage) equals demand.

#### 4.2 Delivery Capacity Constraint
```
Qt[t] ≤ Qt_cap[t]  ∀t ∈ T
```
Delivered quantity cannot exceed production capacity.

#### 4.3 Surplus Definition
```
Et[t] = Qt_cap[t] - Qt[t]  ∀t ∈ T
```
Excess quantity is the difference between capacity and delivery.

#### 4.4 Capacity-Mass Relationship
```
Qt_cap[t] = eta * Mt[t]  ∀t ∈ T
```
Production capacity is proportional to deployed ISRU mass.

#### 4.5 ISRU Mass Balance
```
Mt[0] = M0
Mt[t] = Mt[t-1] + delta_Mt[t]  ∀t ∈ {1,...,T}
```
ISRU mass accumulates over time with new deployments.

#### 4.6 Launch Mass Constraint
```
M_leo[t] = alpha * delta_Mt[t]  ∀t ∈ T
```
Launch mass is proportional to new ISRU deployment.

#### 4.7 Non-negativity Constraints
```
Qt[t], Qt_cap[t], St[t], Et[t], Mt[t], delta_Mt[t], M_leo[t], Q_earth[t] ≥ 0  ∀t ∈ T
```

### 5. Model Characteristics

- **Type:** Mixed-Integer Linear Programming (MILP) when binary decisions are included, Linear Programming (LP) otherwise
- **Time Horizon:** Discrete time periods (typically 5-20 years)
- **Uncertainty:** Handled through scenario-based or stochastic programming approaches
- **Solver Compatibility:** GLPK, CPLEX, Gurobi, CBC

### 6. Solution Interpretation

The optimal solution provides:
- Deployment schedule for ISRU equipment
- Production and delivery plans
- Cost-benefit analysis under uncertainty
- Risk assessment for different demand scenarios

### 7. Model Extensions

The base model can be extended to include:
- Multiple oxygen production technologies
- Reliability and maintenance considerations
- Multi-objective optimization (cost vs. risk)
- Real options for flexible deployment strategies
- Supply chain constraints and logistics

## Implementation Notes

This model is implemented using the Pyomo optimization modeling language in Python, with modular components for variables, constraints, and objectives. The implementation supports multiple solvers and provides detailed solution analysis and visualization capabilities.