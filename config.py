"""
Global configuration and parameters for the entire project.
"""

import numpy as np

# ============================================================
# Random seed for reproducibility
# ============================================================
GLOBAL_SEED = 42

# ============================================================
# Default experiment parameters
# ============================================================
N_SIMULATIONS_DEFAULT = 500  # number of simulation runs per experiment
T_VALUES_DEFAULT = [500, 1000, 2000, 5000, 10000]  # sales horizons to test

# ============================================================
# Single-product experiment (Section 3.1 / Figure 1)
# ============================================================
SINGLE_PRODUCT = {
    'N': 1,                      # number of products
    'M': 1,                      # number of resources
    'K': 4,                      # number of price vectors
    'prices': np.array([[29.90], [34.90], [39.90], [44.90]]),  # [K x N]
    'A': np.array([[1.0]]),      # resource consumption matrix [N x M]
    'alpha_values': [0.25, 0.5], # I = alpha * T
    'mean_demand': np.array([[0.8], [0.6], [0.3], [0.1]]),  # [K x N] Bernoulli prob
    'demand_type': 'bernoulli',
    'prior_type': 'beta',
}

# ============================================================
# Multi-product experiment (Section 3.2 / Figure 2)
# ============================================================
MULTI_PRODUCT = {
    'N': 2,                      # number of products
    'M': 3,                      # number of resources
    'K': 5,                      # number of price vectors
    'prices': np.array([
        [1.0, 1.5],
        [1.0, 2.0],
        [2.0, 3.0],
        [4.0, 4.0],
        [4.0, 6.5],
    ]),                          # [K x N]
    'A': np.array([
        [1.0, 3.0, 0.0],        # product 1 consumption
        [1.0, 1.0, 5.0],        # product 2 consumption
    ]),                          # [N x M]
    'I': np.array([5.0, 5.0, 5.0]),  # initial inventory [M]; scaled by T later
    'demand_types': ['linear', 'exponential', 'logit'],
    'demand_family': 'poisson',
    'prior_type': 'gamma',
}

# ============================================================
# LP solver tolerance
# ============================================================
LP_EPSILON = 1e-8  # small value for numerical stability
PRICE_CLOSE_IDX = None  # sentinel for "closing price" p_inf (None = no sales)

# ============================================================
# Multiplicative constants for UCB/LCB bounds (theory)
# ============================================================
UCB_CONSTANT = 1.0  # can be tuned