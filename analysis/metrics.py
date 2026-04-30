"""
Metrics: summary tables for experiment results.
"""

import numpy as np
from typing import Dict, List


def compute_metrics_table(results: dict) -> List[Dict]:
    """
    Build a summary table from experiment results.

    Args:
        results: nested dict from run_experiment()

    Returns:
        list of dicts, each representing a row:
          {config, algorithm, mean_revenue, std_revenue,
           pct_optimal, n_simulations, T}
    """
    rows = []
    for config_key, entry in results.items():
        baseline = entry.get('baseline', 0.0)
        T_val = entry['config']['T']

        for algo_name, algo_data in entry['algorithms'].items():
            pct = algo_data.get('mean_pct_optimal', 0.0)
            rows.append({
                'config': config_key,
                'algorithm': algo_name,
                'T': T_val,
                'mean_revenue': algo_data['mean_revenue'],
                'std_revenue': algo_data['std_revenue'],
                'pct_optimal': pct,
                'n_simulations': len(algo_data['total_revenues']),
                'baseline': baseline,
            })

    return rows


def print_metrics_table(rows: List[Dict]):
    """Pretty-print metrics table to console."""
    header = f"{'Config':<30} {'Algorithm':<18} {'T':<6} {'Mean Rev':<12} {'Std Rev':<12} {'% Optimal':<10}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for row in rows:
        print(
            f"{row['config']:<30} {row['algorithm']:<18} {row['T']:<6} "
            f"{row['mean_revenue']:<12.2f} {row['std_revenue']:<12.2f} "
            f"{row['pct_optimal']:<10.2f}"
        )


def format_latex_table(rows: List[Dict], caption: str = "Experiment Results") -> str:
    """
    Generate a LaTeX table from metrics rows.

    Args:
        rows: list of dicts from compute_metrics_table()
        caption: table caption

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Algorithm & T & Mean Revenue & Std Dev & \% Optimal \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{row['algorithm']} & {row['T']} & "
            f"{row['mean_revenue']:.2f} & {row['std_revenue']:.2f} & "
            f"{row['pct_optimal']:.2f}\\% \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)