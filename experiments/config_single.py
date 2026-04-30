"""
Configuration for single-product experiments (reproducing Figure 1).

Parameters follow the paper's experimental setup:
  N = 1 (single product)
  M = 1 (single resource)
  K = 4 (four price options)
  Demand: Bernoulli (buy / no-buy)
  Prior: Beta(1, 1) = Uniform[0, 1]
"""

import numpy as np
from typing import Dict, Any, List


def get_single_product_config(
    T: int = 1000,
    inventory_ratio: float = 0.25,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate configuration dict for a single-product experiment.

    Args:
        T: sales horizon length
        inventory_ratio: I / T ratio (alpha in the paper)
        **kwargs: additional overrides

    Returns:
        config dictionary
    """
    N = 1   # one product
    M = 1   # one resource
    K = 4   # four price points

    prices = np.array([[29.90],
                       [34.90],
                       [39.90],
                       [44.90]])  # [K, N]

    # Resource consumption: 1 unit of resource per product
    A = np.ones((N, M))  # [N, M]

    # True mean demand for each price (Bernoulli probability)
    theta_true = np.array([[0.8],
                           [0.6],
                           [0.3],
                           [0.1]])  # [K, N]

    initial_inventory = np.array([inventory_ratio * T])

    config = {
        'N': N,
        'M': M,
        'K': K,
        'T': T,
        'prices': prices,
        'A': A,
        'I': initial_inventory,
        'theta_true': theta_true,
        'inventory_ratio': inventory_ratio,
        'demand_type': 'bernoulli',
        'prior_type': 'beta',
    }
    config.update(kwargs)
    return config


# T values used in the paper
DEFAULT_T_VALUES = [500, 1000, 2000, 5000, 10000]
# Inventory ratios used in the paper
DEFAULT_ALPHA_VALUES = [0.25, 0.5]