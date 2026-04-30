"""
Main entry point for reproducing:
  "Online Network Revenue Management using Thompson Sampling"
  by Ferreira, Simchi-Levi & Wang (2015)

Usage:
    python main.py --experiment single   --n_sim 50    # Figure 1 (quick test)
    python main.py --experiment single   --n_sim 500   # Figure 1 (full)
    python main.py --experiment multi    --n_sim 100   # Figure 2
    python main.py --experiment extended --n_sim 100   # Extended experiments
    python main.py --all                              # Run everything

Output:
    figures/    — generated plots
    results/    — saved experiment data (pickle)
"""

import argparse
import os
import pickle
import sys
import time
from typing import Dict, List, Type

# Ensure non-interactive matplotlib backend for headless/server environments
if os.environ.get('DISPLAY') is None and os.name == 'posix':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np

from analysis.regret import bayesian_regret_summary
from analysis.metrics import compute_metrics_table, print_metrics_table
from analysis.visualizer import (
    plot_single_product_results,
    plot_multi_product_results,
    plot_regret_curve,
    plot_heatmap,
    plot_delta_performance,
)

from experiments.config_single import (
    get_single_product_config,
    DEFAULT_T_VALUES,
    DEFAULT_ALPHA_VALUES,
)
from experiments.config_multi import (
    get_multi_product_config,
    DEFAULT_MULTI_T_VALUES,
    DEMAND_FUNCTIONS,
)
from experiments.runner import run_experiment

from algorithms.ts_fixed import TSFixed
from algorithms.ts_update import TSUpdate
from algorithms.bz import BZAlgorithm
from algorithms.pd_bwk import PDBwK
from algorithms.ts_unconstrained import TSUnconstrained

# Output directories
FIGURES_DIR = "figures"
RESULTS_DIR = "results"


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(data: dict, filename: str):
    """Pickle results to disk."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to {path}")


def load_results(filename: str) -> dict:
    """Load pickled results from disk."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)


# =========================================================================
# Single-product experiments (Figure 1)
# =========================================================================

def run_single_product_experiment(n_simulations: int = 100):
    """Reproduce Figure 1: single-product, Bernoulli demand."""
    print("=" * 60)
    print("Single-Product Experiment (Figure 1)")
    print("=" * 60)

    algo_classes: Dict[str, Type] = {
        'TS-fixed': TSFixed,
        'TS-update': TSUpdate,
        'BZ': BZAlgorithm,
        'PD-BwK': PDBwK,
        'TS (unconstrained)': TSUnconstrained,
    }

    all_results = {}

    for alpha in DEFAULT_ALPHA_VALUES:
        print(f"\n--- Inventory ratio alpha = {alpha} ---")

        # Build list of configs for each T value
        configs = [
            get_single_product_config(T=T, inventory_ratio=alpha)
            for T in DEFAULT_T_VALUES
        ]

        start_time = time.time()
        results = run_experiment(configs, algo_classes, n_simulations)
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")

        key = f"alpha={alpha}"
        all_results[key] = results

        # Save per-alpha results
        save_results(results, f"single_alpha{alpha}.pkl")

    # Plot Figure 1
    algo_names = list(algo_classes.keys())
    plot_single_product_results(
        all_results,
        algo_names,
        save_path=os.path.join(FIGURES_DIR, "figure1_single_product.png"),
    )

    # Print summary table
    for alpha_key, results in all_results.items():
        print(f"\n--- Summary: {alpha_key} ---")
        rows = compute_metrics_table(results)
        print_metrics_table(rows)

    return all_results


# =========================================================================
# Multi-product experiments (Figure 2)
# =========================================================================

def run_multi_product_experiment(n_simulations: int = 100):
    """Reproduce Figure 2: multi-product, Poisson demand."""
    print("\n" + "=" * 60)
    print("Multi-Product Experiment (Figure 2)")
    print("=" * 60)

    # Multi-product algorithms
    algo_classes: Dict[str, Type] = {
        'TS-fixed': TSFixed,
        'TS-update': TSUpdate,
        'BZ': BZAlgorithm,
        # PD-BwK not applied to Poisson (per paper)
    }

    all_results = {}

    for demand_type in ['linear', 'exponential', 'logit']:
        print(f"\n--- Demand type: {demand_type} ---")

        configs = [
            get_multi_product_config(T=T, demand_type=demand_type)
            for T in DEFAULT_MULTI_T_VALUES
        ]

        start_time = time.time()
        results = run_experiment(configs, algo_classes, n_simulations)
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")

        key = demand_type
        all_results[key] = results
        save_results(results, f"multi_{demand_type}.pkl")

    # Plot Figure 2
    algo_names = list(algo_classes.keys())
    plot_multi_product_results(
        all_results,
        algo_names,
        save_path=os.path.join(FIGURES_DIR, "figure2_multi_product.png"),
    )

    for demand_key, results in all_results.items():
        print(f"\n--- Summary: {demand_key} ---")
        rows = compute_metrics_table(results)
        print_metrics_table(rows)

    return all_results


# =========================================================================
# Extended experiments (additional figures)
# =========================================================================

def run_extended_experiments(n_simulations: int = 100):
    """Run extended experiments (Figures 3-5, Table 1)."""
    print("\n" + "=" * 60)
    print("Extended Experiments")
    print("=" * 60)

    algo_classes: Dict[str, Type] = {
        'TS-fixed': TSFixed,
        'TS-update': TSUpdate,
        'BZ': BZAlgorithm,
    }

    # ---- Figure 3: Regret curve (log-log) ----
    print("\n--- Figure 3: Regret Curve ---")
    config = get_single_product_config(T=10000, inventory_ratio=0.25)
    results = run_experiment([config], algo_classes, n_simulations, track_period_revenues=True)

    config_key = list(results.keys())[0]
    entry = results[config_key]
    baseline = entry['baseline']
    baseline_per_period = baseline / config['T']

    # Build regret summary
    per_period_data = {}
    for algo_name, algo_data in entry['algorithms'].items():
        per_period_data[algo_name] = algo_data['per_period_revenues']

    regret_summary = bayesian_regret_summary(per_period_data, baseline_per_period)

    plot_regret_curve(
        regret_summary,
        T=config['T'],
        save_path=os.path.join(FIGURES_DIR, "figure3_regret_curve.png"),
    )

    # ---- Figure 4: Heatmap ----
    print("\n--- Figure 4: Performance Heatmap ---")
    alpha_values = np.linspace(0.1, 0.9, 9)
    t_values = [500, 1000, 2000, 5000]

    perf_matrix = np.zeros((len(alpha_values), len(t_values)))

    for i, alpha in enumerate(alpha_values):
        configs = [
            get_single_product_config(T=T, inventory_ratio=alpha)
            for T in t_values
        ]
        h_results = run_experiment(configs, {'TS-update': TSUpdate}, n_simulations // 2)

        for j, T_val in enumerate(t_values):
            key = f"T={T_val}_inv={alpha}_demand=bernoulli"
            if key in h_results:
                algo_data = h_results[key]['algorithms'].get('TS-update', {})
                pct = algo_data.get('mean_pct_optimal', 0.0)
                perf_matrix[i, j] = pct

    plot_heatmap(
        alpha_values=list(alpha_values),
        t_values=t_values,
        performance_matrix=perf_matrix,
        save_path=os.path.join(FIGURES_DIR, "figure4_heatmap.png"),
    )

    # ---- Figure 5: Delta performance TS-update vs TS-fixed ----
    print("\n--- Figure 5: Delta Performance (TS-update - TS-fixed) ---")
    fixed_T = 1000
    alpha_values_scan = np.linspace(0.1, 0.9, 9)
    delta_values = []

    for alpha in alpha_values_scan:
        configs = [get_single_product_config(T=fixed_T, inventory_ratio=alpha)]
        d_results = run_experiment(configs, algo_classes, n_simulations // 2)

        key = list(d_results.keys())[0]
        entry = d_results[key]

        pct_fixed = entry['algorithms'].get('TS-fixed', {}).get('mean_pct_optimal', 0.0)
        pct_update = entry['algorithms'].get('TS-update', {}).get('mean_pct_optimal', 0.0)
        delta_values.append(pct_update - pct_fixed)

    plot_delta_performance(
        alpha_values=list(alpha_values_scan),
        delta_values=delta_values,
        T=fixed_T,
        save_path=os.path.join(FIGURES_DIR, "figure5_delta_performance.png"),
    )

    # ---- Table 1: Final performance at T=10000 ----
    print("\n--- Table 1: Performance at T=10000 ---")
    t_final = 10000
    single_config = get_single_product_config(T=t_final, inventory_ratio=0.25)
    multi_config_linear = get_multi_product_config(T=t_final, demand_type='linear')

    final_results = run_experiment(
        [single_config, multi_config_linear],
        algo_classes,
        max(n_simulations // 2, 10),
    )

    rows = compute_metrics_table(final_results)
    print_metrics_table(rows)

    return {
        'regret_summary': regret_summary,
        'heatmap': perf_matrix,
        'delta': (list(alpha_values_scan), delta_values),
        'table': rows,
    }


# =========================================================================
# Main entry point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduction: Online Network Revenue Management using Thompson Sampling"
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['single', 'multi', 'extended'],
        help='Which experiment to run'
    )
    parser.add_argument(
        '--n_sim', '-n',
        type=int,
        default=50,
        help='Number of simulations (default: 50 for quick test; paper uses 500)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all experiments'
    )
    args = parser.parse_args()

    ensure_dirs()

    # Display header
    print("=" * 60)
    print("Reproducing: Online Network Revenue Management")
    print("             using Thompson Sampling")
    print("             Ferreira, Simchi-Levi & Wang (2015)")
    print("=" * 60)
    print(f"Simulations per config: {args.n_sim}")
    print()

    if args.all or args.experiment == 'single':
        run_single_product_experiment(args.n_sim)

    if args.all or args.experiment == 'multi':
        run_multi_product_experiment(args.n_sim)

    if args.all or args.experiment == 'extended':
        run_extended_experiments(args.n_sim)

    print("\n" + "=" * 60)
    print("All experiments complete. Figures saved to 'figures/'.")
    print("=" * 60)


if __name__ == '__main__':
    main()