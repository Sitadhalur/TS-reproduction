"""
Regret computation and analysis tools.
"""

import numpy as np
from typing import List


def compute_regret(
    algorithm_revenues: np.ndarray,
    optimal_revenue: float,
) -> np.ndarray:
    """
    Compute Bayesian Regret for each simulation.

    Args:
        algorithm_revenues: [n_simulations] array of total revenues
        optimal_revenue: optimal expected total revenue (from LP with true params)

    Returns:
        regret: [n_simulations] array of regret values
    """
    return np.maximum(0.0, optimal_revenue - algorithm_revenues)


def cumulative_regret_curve(
    per_period_revenues: List[np.ndarray],
    optimal_per_period_revenue: float,
) -> np.ndarray:
    """
    Compute the cumulative regret curve averaged over simulations.

    Args:
        per_period_revenues: list of [T] arrays, one per simulation
        optimal_per_period_revenue: optimal expected revenue per period

    Returns:
        [T] array of mean cumulative regret at each period
    """
    T = len(per_period_revenues[0])
    n_sim = len(per_period_revenues)

    cumulative_regrets = np.zeros((n_sim, T))
    for i, revenues in enumerate(per_period_revenues):
        cumulative_opt = np.arange(1, T + 1, dtype=float) * optimal_per_period_revenue
        cumulative_rev = np.cumsum(revenues)
        cumulative_regrets[i] = cumulative_opt - cumulative_rev

    return np.mean(cumulative_regrets, axis=0)


def bayesian_regret_summary(
    per_period_revenues_all: dict,
    optimal_per_period: float,
) -> dict:
    """
    Compute Bayesian regret summary for multiple algorithms.

    Args:
        per_period_revenues_all: {algo_name: [list of [T] arrays]}
        optimal_per_period: optimal expected revenue per period

    Returns:
        {algo_name: {
            'regret_curve': [T] array of mean cumulative regret,
            'final_regret_mean': float,
            'final_regret_std': float
        }}
    """
    summary = {}
    T = None
    for algo_name, rev_list in per_period_revenues_all.items():
        if T is None and len(rev_list) > 0:
            T = len(rev_list[0])
        curve = cumulative_regret_curve(rev_list, optimal_per_period)
        final_regrets = []
        for rev in rev_list:
            total_opt = len(rev) * optimal_per_period
            total_algo = np.sum(rev)
            final_regrets.append(max(0.0, total_opt - total_algo))

        summary[algo_name] = {
            'regret_curve': curve,
            'final_regret_mean': float(np.mean(final_regrets)),
            'final_regret_std': float(np.std(final_regrets, ddof=1)),
            'T': T,
        }
    return summary