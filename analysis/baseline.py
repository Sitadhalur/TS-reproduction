"""
Baseline computation: optimal expected revenue using true demand parameters.

Solves the LP with true theta to get OPT(d), the maximum expected per-period revenue.
This is used as the benchmark for computing % of optimal revenue.
"""

import numpy as np
from utils.lp_solver import solve_pricing_lp


def compute_optimal_revenue(config: dict) -> float:
    """
    Compute the optimal expected total revenue OPT(d) * T using true demand parameters.

    Solves:
      max  Σ_k (Σ_i p_{ik} * d_{ik}) * x_k
      s.t. Σ_k (Σ_i a_{ij} * d_{ik}) * x_k ≤ c_j,  ∀j
           Σ_k x_k ≤ 1

    Args:
        config: experiment configuration dictionary

    Returns:
        optimal_total_revenue: float
    """
    prices = config['prices']           # [K, N]
    theta_true = config['theta_true']   # [K, N]
    A = config['A']                     # [N, M]
    I = config['I']                     # [M]
    T = config['T']

    # Capacity rates c_j = I_j / T
    c = I / T

    # Solve LP with true demand
    # theta_true is [K, N] — matches solver's expected (K, N) shape
    x_opt, _ = solve_pricing_lp(prices, theta_true, A, c)

    if x_opt is None or not np.any(x_opt > 1e-10):
        # Fallback: if LP is infeasible, use best single price
        revenues = np.sum(prices * theta_true, axis=1)
        best_revenue = np.max(revenues)
        return best_revenue * T

    # Optimal per-period revenue
    revenues = np.sum(prices * theta_true, axis=1)  # [K]
    opt_per_period = float(np.dot(revenues, x_opt))

    return opt_per_period * T
