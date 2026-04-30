"""
Experiment runner: ties together environment, algorithms, and evaluation.

Handles:
  - Single simulation run
  - Batch experiment runs with multiple algorithms
  - Per-period revenue tracking for regret curves
  - Parallel execution via joblib for embarrassingly parallel simulations
"""

import numpy as np
from typing import Dict, List, Optional, Type, Callable
from tqdm import tqdm

from models.revenue_network import RevenueNetwork
from models.demand import DemandModel, BernoulliDemand, PoissonDemand
from algorithms.base import DynamicPricingAlgorithm
from .config_single import get_single_product_config
from .config_multi import get_multi_product_config
from analysis.baseline import compute_optimal_revenue


def run_single_simulation(
    config: dict,
    algorithm: DynamicPricingAlgorithm,
    env_seed: int,
    algo_seed: int,
    track_period_revenues: bool = False,
) -> tuple:
    """
    Run one complete simulation over T periods.

    Args:
        config: experiment configuration dictionary
        algorithm: initialised algorithm instance (will be re-initialised)
        env_seed: seed for environment randomness (demand realisations)
        algo_seed: seed for algorithm randomness (sampling)
        track_period_revenues: if True, return per-period revenue array;
            if False, return None (saves memory and disk space)

    Returns:
        (total_revenue: float, per_period_revenues: np.ndarray [T] or None)
    """
    T = config['T']
    N = config['N']
    M = config['M']
    K = config['K']
    prices = config['prices']
    A = config['A']
    I = config['I']
    theta_true = config['theta_true']
    demand_type = config.get('demand_type', 'bernoulli')

    # Create demand model
    if demand_type == 'bernoulli':
        demand_model = BernoulliDemand(theta_true)
    elif demand_type == 'poisson':
        # For Poisson, theta_true is the mean demand at each price
        demand_fn = config.get('demand_function', 'linear')
        demand_model = PoissonDemand(mean_function=demand_fn, prices=prices)
    else:
        raise ValueError(f"Unknown demand type: {demand_type}")

    # Create environment
    env = RevenueNetwork(
        N=N, M=M, K=K,
        prices=prices,
        consumption_matrix=A,
        initial_inventory=I,
        T=T,
        demand_model=demand_model,
        theta_true=theta_true,
        seed=env_seed,
    )
    env.reset()

    # Initialise algorithm
    algorithm.rng = np.random.default_rng(algo_seed)
    algorithm.initialize(env)

    # Run T periods
    total_revenue = 0.0
    if track_period_revenues:
        per_period_revenues = np.zeros(T)
    else:
        per_period_revenues = None

    for t in range(1, T + 1):
        # Track current period in env (used by TS-update's get_remaining_capacity_rate)
        env._period = t

        # Choose price
        price_idx = algorithm.choose_price(t, env)

        # Execute
        demand, revenue, stock_out = env.step(price_idx)

        # Update algorithm
        algorithm.update(t, price_idx, demand)

        # Record
        total_revenue += revenue
        if track_period_revenues:
            per_period_revenues[t - 1] = revenue

    return total_revenue, per_period_revenues


def run_experiment(
    configs: List[dict],
    algorithm_classes: Dict[str, Type[DynamicPricingAlgorithm]],
    n_simulations: int = 100,
    show_progress: bool = True,
    compute_baseline: bool = True,
    n_jobs: int = -1,
    track_period_revenues: bool = False,
) -> dict:
    """
    Run a full experiment across multiple configurations and algorithms.

    Uses joblib.Parallel for embarrassingly parallel simulation runs.
    On a multi-core machine (e.g., AMD Ryzen 9 7945HX, 16 cores),
    this achieves ~12-15x speedup over serial execution.

    Args:
        configs: list of config dicts (each a different T value or setting)
        algorithm_classes: {name: class} mapping
        n_simulations: number of simulation runs per (config, algorithm)
        show_progress: show tqdm progress bar
        compute_baseline: if True, compute optimal revenue as benchmark
        n_jobs: number of parallel jobs (-1 = use all available cores)

    Returns:
        results: nested dict structure:
            {config_key: {
                'config': config,
                'baseline': float (optimal revenue),
                'algorithms': {algo_name: {
                    'total_revenues': [float],
                    'mean_revenue': float,
                    'std_revenue': float,
                    'mean_pct_optimal': float,
                    'per_period_revenues': [np.ndarray] (if tracked)
                }}
            }}
    """
    # Try to import joblib; fall back to serial if not available
    try:
        from joblib import Parallel, delayed
        _PARALLEL_AVAILABLE = True
    except ImportError:
        _PARALLEL_AVAILABLE = False

    results = {}

    iterator = tqdm(configs, desc='Configurations') if show_progress else configs

    for cfg_idx, config in enumerate(iterator):
        # Create a unique key for this configuration
        T = config['T']
        inv_ratio = config.get('inventory_ratio', config.get('inventory_scaling', '?'))
        demand_fn = config.get('demand_function', 'bernoulli')
        config_key = f"T={T}_inv={inv_ratio}_demand={demand_fn}"

        entry = {
            'config': config,
            'algorithms': {},
        }

        # Compute optimal baseline revenue
        if compute_baseline:
            entry['baseline'] = compute_optimal_revenue(config)
            entry['baseline_per_period'] = entry['baseline'] / T

        # Run simulations for each algorithm
        for algo_name, algo_cls in algorithm_classes.items():
            total_revenues = []
            all_per_period = [] if track_period_revenues else None

            if show_progress and len(configs) == 1:
                sim_range = tqdm(range(n_simulations), desc=f'{algo_name}', leave=False)
            else:
                sim_range = range(n_simulations)

            if _PARALLEL_AVAILABLE and n_jobs != 1:
                # === PARALLEL EXECUTION (joblib) ===
                # Simulations are embarrassingly parallel: each sim is independent
                # Seed separation: env_seed != algo_seed ensures decorrelated randomness
                def run_one_sim(sim: int):
                    env_seed = cfg_idx * n_simulations * 100 + sim
                    algo_seed = cfg_idx * n_simulations * 100 + sim + 50000
                    algo = algo_cls(config)
                    return run_single_simulation(config, algo, env_seed, algo_seed, track_period_revenues)

                # n_jobs=-1 uses all available CPU cores
                # backend='loky' works reliably on Windows
                parallel_results = Parallel(n_jobs=n_jobs, prefer='processes')(
                    delayed(run_one_sim)(sim) for sim in sim_range
                )

                for total_rev, per_period in parallel_results:
                    total_revenues.append(total_rev)
                    if track_period_revenues:
                        all_per_period.append(per_period)
            else:
                # === SERIAL EXECUTION (fallback) ===
                for sim in sim_range:
                    env_seed = cfg_idx * n_simulations * 100 + sim
                    algo_seed = cfg_idx * n_simulations * 100 + sim + 50000

                    algo = algo_cls(config)
                    total_rev, per_period = run_single_simulation(
                        config, algo, env_seed, algo_seed, track_period_revenues
                    )
                    total_revenues.append(total_rev)
                    if track_period_revenues:
                        all_per_period.append(per_period)

            # Aggregate
            total_arr = np.array(total_revenues)
            mean_rev = float(np.mean(total_arr))
            std_rev = float(np.std(total_arr, ddof=1))

            algo_entry = {
                'total_revenues': total_arr,
                'mean_revenue': mean_rev,
                'std_revenue': std_rev,
            }
            if track_period_revenues:
                algo_entry['per_period_revenues'] = all_per_period

            if 'baseline' in entry and entry['baseline'] > 0:
                algo_entry['mean_pct_optimal'] = \
                    (mean_rev / entry['baseline']) * 100.0
            else:
                algo_entry['mean_pct_optimal'] = 0.0

            entry['algorithms'][algo_name] = algo_entry

        results[config_key] = entry

    return results