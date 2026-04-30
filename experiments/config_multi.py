"""
Configuration for multi-product experiments (reproducing Figure 2).

Parameters follow the paper's experimental setup:
  N = 2 (two products)
  M = 3 (three resources)
  K = 5 (five price vectors)
  Demand: Poisson with three demand function types (linear, exponential, logit)
  Prior: Gamma(1, 1)
"""

import numpy as np
from typing import Dict, Any, Callable


# --- Consumption matrix ---
# Product 1: consumes (1, 3, 0) units of resources (1, 2, 3)
# Product 2: consumes (1, 1, 5) units of resources (1, 2, 3)
CONSUMPTION_MATRIX = np.array([[1, 3, 0],
                                [1, 1, 5]])  # [N, M]

# Price vectors [K, N]: (p1, p2)
PRICE_VECTORS = np.array([
    [1.0,  1.5],
    [1.0,  2.0],
    [2.0,  3.0],
    [4.0,  4.0],
    [4.0,  6.5],
])


def linear_demand(prices: np.ndarray) -> np.ndarray:
    """
    Linear demand: mu(p) = (8 - 1.5*p1, 9 - 3*p2)
    Args:
        prices: [K, N] price matrix
    Returns:
        [K, N] mean demand for each price vector
    """
    mu = np.zeros_like(prices)
    mu[:, 0] = 8.0 - 1.5 * prices[:, 0]
    mu[:, 1] = 9.0 - 3.0 * prices[:, 1]
    return np.maximum(mu, 0.0)  # non-negative


def exponential_demand(prices: np.ndarray) -> np.ndarray:
    """
    Exponential demand: mu(p) = (5 * exp(-0.5*p1), 9 * exp(-p2))
    """
    mu = np.zeros_like(prices)
    mu[:, 0] = 5.0 * np.exp(-0.5 * prices[:, 0])
    mu[:, 1] = 9.0 * np.exp(-prices[:, 1])
    return mu


def logit_demand(prices: np.ndarray) -> np.ndarray:
    """
    Logit demand (discrete choice model):
      mu1 = 10 * exp(-p1) / (1 + exp(-p1) + exp(-p2))
      mu2 = 10 * exp(-p2) / (1 + exp(-p1) + exp(-p2))
    """
    exp_p1 = np.exp(-prices[:, 0])
    exp_p2 = np.exp(-prices[:, 1])
    denominator = 1.0 + exp_p1 + exp_p2
    mu = np.zeros_like(prices)
    mu[:, 0] = 10.0 * exp_p1 / denominator
    mu[:, 1] = 10.0 * exp_p2 / denominator
    return mu


DEMAND_FUNCTIONS: Dict[str, Callable] = {
    'linear': linear_demand,
    'exponential': exponential_demand,
    'logit': logit_demand,
}


def get_multi_product_config(
    T: int = 1000,
    demand_type: str = 'linear',
    inventory_scaling: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate configuration dict for a multi-product experiment.

    Args:
        T: sales horizon length
        demand_type: 'linear', 'exponential', or 'logit'
        inventory_scaling: multiply base inventory by this factor
        **kwargs: additional overrides

    Returns:
        config dictionary
    """
    N = 2
    M = 3
    K = 5

    prices = PRICE_VECTORS.copy()
    A = CONSUMPTION_MATRIX.copy()

    # True mean demand for each price vector under the chosen demand function
    demand_fn = DEMAND_FUNCTIONS[demand_type]
    theta_true = demand_fn(prices)  # [K, N]

    # Per-period capacity rates c = I / T.
    # The paper's multi-product setting uses inventory levels proportional to T,
    # so I should be O(T), not O(T / 100). The previous implementation divided
    # by 100, which made the inventory 100x too small and caused the multi-
    # product benchmarks (especially BZ) to fail abnormally.
    base_capacity_rates = np.asarray(
        kwargs.pop('base_capacity_rates', np.array([3.0, 5.0, 7.0], dtype=float)),
        dtype=float,
    ) * inventory_scaling
    initial_inventory = base_capacity_rates * float(T)

    config = {
        'N': N,
        'M': M,
        'K': K,
        'T': T,
        'prices': prices,
        'A': A,
        'I': initial_inventory,
        'theta_true': theta_true,
        'base_capacity_rates': base_capacity_rates,
        'demand_type': 'poisson',
        'prior_type': 'gamma',
        'demand_function': demand_type,
        'inventory_scaling': inventory_scaling,
    }
    config.update(kwargs)
    return config


DEFAULT_MULTI_T_VALUES = [500, 1000, 2000, 5000, 10000]