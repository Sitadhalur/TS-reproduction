"""
Visualisation tools for experiment results.

Generates publication-quality figures reproducing the paper's style.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Global style settings
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'figure.figsize': (8, 5),
})

# Colorblind-friendly palette
COLORS = {
    'TS-fixed': '#1f77b4',       # blue
    'TS-update': '#ff7f0e',      # orange
    'BZ': '#2ca02c',             # green
    'PD-BwK': '#d62728',         # red
    'TS (unconstrained)': '#9467bd',  # purple
    'TS-linear': '#8c564b',      # brown
    'TS-contextual': '#7f7f7f',  # gray
    'TS-BwK': '#bcbd22',         # yellow-green
}

MARKERS = {
    'TS-fixed': 'o',
    'TS-update': 's',
    'BZ': 'D',
    'PD-BwK': '^',
    'TS (unconstrained)': 'v',
    'TS-linear': 'P',
    'TS-contextual': 'X',
    'TS-BwK': '*',
}


def _extract_pct_optimal(results: dict, algo_names: List[str]) -> dict:
    """Extract % optimal for each algorithm and T value."""
    data = {name: [] for name in algo_names}
    t_values = []

    # Sort config keys by T
    config_keys = sorted(results.keys(),
                          key=lambda k: int(k.split('_')[0].split('=')[1]))

    for key in config_keys:
        entry = results[key]
        T_val = entry['config']['T']
        t_values.append(T_val)

        for name in algo_names:
            if name in entry['algorithms']:
                pct = entry['algorithms'][name].get('mean_pct_optimal', 0.0)
                data[name].append(pct)
            else:
                data[name].append(np.nan)

    return data, t_values


def plot_single_product_results(
    results_dict: Dict[str, dict],
    algo_names: List[str],
    title: str = "Single Product Experiment",
    save_path: Optional[str] = None,
):
    """
    Plot Figure 1: % of optimal revenue vs. T for single-product experiments.

    Args:
        results_dict: {alpha_key: results_from_run_experiment()}
                      where alpha_key is e.g. 'alpha=0.25' or 'alpha=0.5'
        algo_names: list of algorithm names to include
        title: plot title
        save_path: path to save the figure
    """
    n_alphas = len(results_dict)
    fig, axes = plt.subplots(1, n_alphas, figsize=(6 * n_alphas, 5),
                             sharey=True)

    if n_alphas == 1:
        axes = [axes]

    for idx, (alpha_key, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        data, t_values = _extract_pct_optimal(results, algo_names)

        for name in algo_names:
            vals = data[name]
            if any(np.isfinite(v) for v in vals):
                ax.plot(t_values, vals,
                        marker=MARKERS.get(name, 'o'),
                        label=name,
                        color=COLORS.get(name, None),
                        markersize=6)

        ax.set_xlabel('T (Sales Horizon)')
        if idx == 0:
            ax.set_ylabel('% of Optimal Revenue')
        ax.set_title(f'{alpha_key}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

        # Log scale for x-axis
        ax.set_xscale('log')

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def plot_multi_product_results(
    results_dict: Dict[str, dict],
    algo_names: List[str],
    title: str = "Multi-Product Experiment",
    save_path: Optional[str] = None,
):
    """
    Plot Figure 2: % of optimal revenue vs. T for multi-product experiments.

    Args:
        results_dict: {demand_type_key: results_from_run_experiment()}
        algo_names: list of algorithm names to include
        title: plot title
        save_path: path to save the figure
    """
    n_types = len(results_dict)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5),
                             sharey=True)

    if n_types == 1:
        axes = [axes]

    for idx, (demand_key, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        data, t_values = _extract_pct_optimal(results, algo_names)

        for name in algo_names:
            vals = data[name]
            if any(np.isfinite(v) for v in vals):
                ax.plot(t_values, vals,
                        marker=MARKERS.get(name, 'o'),
                        label=name,
                        color=COLORS.get(name, None),
                        markersize=6)

        ax.set_xlabel('T (Sales Horizon)')
        if idx == 0:
            ax.set_ylabel('% of Optimal Revenue')
        ax.set_title(f'Demand: {demand_key}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        ax.set_xscale('log')

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def plot_regret_curve(
    regret_summary: dict,
    T: int,
    title: str = "Cumulative Bayesian Regret",
    save_path: Optional[str] = None,
):
    """
    Plot cumulative regret curve over time.

    Args:
        regret_summary: {algo_name: {'regret_curve': [T] array}} from bayesian_regret_summary()
        T: horizon length
        title: plot title
        save_path: path to save the figure
    """
    fig, ax = plt.subplots()

    for algo_name, data in regret_summary.items():
        curve = data['regret_curve']
        ax.plot(np.arange(1, T + 1), curve,
                label=algo_name,
                color=COLORS.get(algo_name, None))

    # Add reference line for sqrt(T)
    ts = np.arange(1, T + 1)
    sqrt_ref = np.sqrt(ts) * (regret_summary[list(regret_summary.keys())[0]]['final_regret_mean'] / np.sqrt(T))
    ax.plot(ts, sqrt_ref, '--k', alpha=0.5, label=f'O(sqrt(T))')

    ax.set_xlabel('Period t')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def plot_heatmap(
    alpha_values: List[float],
    t_values: List[int],
    performance_matrix: np.ndarray,
    title: str = "Performance Heatmap",
    xlabel: str = "T (Sales Horizon)",
    ylabel: str = r"$\alpha$ (Inventory Ratio)",
    save_path: Optional[str] = None,
):
    """
    Plot a heatmap of performance across (alpha, T) grid.

    Args:
        alpha_values: list of inventory ratios
        t_values: list of T values
        performance_matrix: [n_alpha, n_T] matrix of % optimal revenue
        title: plot title
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(performance_matrix, aspect='auto', cmap='viridis',
                   interpolation='nearest')

    # Labels
    ax.set_xticks(np.arange(len(t_values)))
    ax.set_xticklabels([str(t) for t in t_values])
    ax.set_yticks(np.arange(len(alpha_values)))
    ax.set_yticklabels([f'{a:.2f}' for a in alpha_values])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('% of Optimal Revenue')

    # Annotate cells
    for i in range(len(alpha_values)):
        for j in range(len(t_values)):
            val = performance_matrix[i, j]
            text_color = 'white' if val < 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    color=text_color, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def plot_delta_performance(
    alpha_values: List[float],
    delta_values: List[float],
    T: int = 1000,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Plot TS-update minus TS-fixed performance difference vs. alpha.

    Positive delta means TS-update outperforms TS-fixed.

    Args:
        alpha_values: list of inventory ratios
        delta_values: list of performance differences (in % points)
        T: the fixed T value used
        save_path: path to save the figure
    """
    if title is None:
        title = f"TS-update vs TS-fixed (T={T})"

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(alpha_values, delta_values, 'o-', color='#1f77b4',
            markersize=8, linewidth=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7,
               label='Zero difference')

    ax.set_xlabel(r'$\alpha$ (Inventory Ratio I/T)')
    ax.set_ylabel(r'$\Delta$ (% of Optimal Revenue)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate each point
    for i, (alpha, delta) in enumerate(zip(alpha_values, delta_values)):
        label = f'{delta:+.1f}%'
        ax.annotate(label, (alpha, delta),
                    textcoords="offset points",
                    xytext=(0, 10 if delta >= 0 else -15),
                    ha='center', fontsize=9)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)