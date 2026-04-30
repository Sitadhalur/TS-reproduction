"""
Statistical utilities for simulation results analysis.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy import stats as scipy_stats


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute mean and confidence interval for an array of observations.

    Args:
        data: array of results (e.g., revenues across simulations)
        confidence: confidence level (default 0.95)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    n = len(data)
    mean = float(np.mean(data))
    if n <= 1:
        return mean, mean, mean

    sem = float(scipy_stats.sem(data))  # standard error of the mean
    h = sem * scipy_stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return mean, mean - h, mean + h


def aggregate_results(
    raw_results: Dict[str, List[float]],
    confidence: float = 0.95,
) -> Dict[str, dict]:
    """
    Aggregate raw simulation results into summary statistics.

    Args:
        raw_results: {algorithm_name: [list of total revenues per simulation]}
        confidence: confidence level

    Returns:
        {algorithm_name: {
            'mean': float,
            'std': float,
            'ci_lower': float,
            'ci_upper': float,
            'median': float,
            'n_simulations': int
        }}
    """
    aggregated = {}
    for name, revenues in raw_results.items():
        arr = np.asarray(revenues)
        mean, ci_low, ci_high = compute_confidence_interval(arr, confidence)
        aggregated[name] = {
            'mean': mean,
            'std': float(np.std(arr, ddof=1)),
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'median': float(np.median(arr)),
            'n_simulations': len(arr),
        }
    return aggregated


def compute_regret(
    algorithm_revenues: List[float],
    optimal_revenue: float,
) -> np.ndarray:
    """
    Compute regret for each simulation.

    Args:
        algorithm_revenues: list of total revenue per simulation
        optimal_revenue: optimal expected revenue (from LP with true params)

    Returns:
        array of regret values (positive = revenue shortfall)
    """
    return np.maximum(0.0, optimal_revenue - np.asarray(algorithm_revenues))


def cumulative_regret_curve(
    per_period_revenues: List[np.ndarray],
    optimal_per_period: float,
) -> np.ndarray:
    """
    Compute cumulative regret curve over time.

    Args:
        per_period_revenues: list of arrays, each [T] revenue per period per simulation
        optimal_per_period: optimal expected revenue PER PERIOD

    Returns:
        [T] mean cumulative regret curve
    """
    T = len(per_period_revenues[0])
    n_sim = len(per_period_revenues)

    regret_curves = np.zeros((n_sim, T))
    for i, revenues in enumerate(per_period_revenues):
        cumulative_opt = np.arange(1, T + 1) * optimal_per_period
        cumulative_algo = np.cumsum(revenues)
        regret_curves[i] = cumulative_opt - cumulative_algo

    return np.mean(regret_curves, axis=0)