"""
Benchmark script: measure simulation time on this device and extrapolate full run time.

Tests: single product, T=1000, 10 simulations, all algorithms.
Also tests T=10000 to measure LP scaling.
Also tests multi-product at T=1000.
Then extrapolates to full experiment sizes with T-segmented weighting.
"""

import time
import numpy as np
from experiments.config_single import get_single_product_config, DEFAULT_T_VALUES, DEFAULT_ALPHA_VALUES
from experiments.config_multi import get_multi_product_config, DEFAULT_MULTI_T_VALUES
from experiments.runner import run_experiment
from algorithms.ts_fixed import TSFixed
from algorithms.ts_update import TSUpdate
from algorithms.bz import BZAlgorithm
from algorithms.pd_bwk import PDBwK
from algorithms.ts_unconstrained import TSUnconstrained

# =========================================================================
# Step 1: Single product baseline at T=1000
# =========================================================================
config = get_single_product_config(T=1000, inventory_ratio=0.25)

algorithm_classes_single = {
    'TS-fixed': TSFixed,
    'TS-update': TSUpdate,
    'BZ': BZAlgorithm,
    'PD-BwK': PDBwK,
    'TS-unconstrained': TSUnconstrained,
}
algo_count_single = len(algorithm_classes_single)

n_test = 10

print(f"Benchmarking: single product, T=1000, {n_test} sims x {algo_count_single} algorithms")
print(f"Config: N={config['N']}, M={config['M']}, K={config['K']}, T={config['T']}")
print()

start = time.time()
results = run_experiment(
    configs=[config],
    algorithm_classes=algorithm_classes_single,
    n_simulations=n_test,
    show_progress=True,
    compute_baseline=True,
)
elapsed = time.time() - start
total_sims = n_test * algo_count_single
time_per_sim_1000 = elapsed / total_sims

print(f"\n=== SINGLE PRODUCT BASELINE ===")
print(f"Total time for {n_test} sims x {algo_count_single} algos: {elapsed:.1f}s ({elapsed/60:.2f} min)")
print(f"Time per single simulation (T=1000): {time_per_sim_1000:.4f}s")

# =========================================================================
# Step 2: T=10000 scaling test (single product)
# =========================================================================
print(f"\n--- T scaling test: T=10000 ---")
config_large = get_single_product_config(T=10000, inventory_ratio=0.25)
start2 = time.time()
results2 = run_experiment(
    configs=[config_large],
    algorithm_classes={'TS-fixed': TSFixed},
    n_simulations=3,
    show_progress=False,
    compute_baseline=True,
)
elapsed2 = time.time() - start2
time_per_sim_10000 = elapsed2 / 3
scaling_ratio = time_per_sim_10000 / time_per_sim_1000

print(f"T=10000, 3 sims x 1 algo: {elapsed2:.1f}s")
print(f"Time per sim at T=10000: {time_per_sim_10000:.4f}s (vs {time_per_sim_1000:.4f}s at T=1000)")
print(f"Scaling factor: {scaling_ratio:.1f}x")

# =========================================================================
# Step 3: Multi-product baseline at T=1000
# =========================================================================
print(f"\n--- Multi-product test: T=1000 ---")
config_multi = get_multi_product_config(T=1000, demand_type='linear')

algorithm_classes_multi = {
    'TS-fixed': TSFixed,
    'TS-update': TSUpdate,
    'BZ': BZAlgorithm,
}
algo_count_multi = len(algorithm_classes_multi)

start3 = time.time()
results3 = run_experiment(
    configs=[config_multi],
    algorithm_classes=algorithm_classes_multi,
    n_simulations=n_test,
    show_progress=True,
    compute_baseline=True,
)
elapsed3 = time.time() - start3
total_sims_multi = n_test * algo_count_multi
time_per_sim_multi_1000 = elapsed3 / total_sims_multi
multi_overhead = time_per_sim_multi_1000 / time_per_sim_1000

print(f"Multi-product: {n_test} sims x {algo_count_multi} algos: {elapsed3:.1f}s ({elapsed3/60:.2f} min)")
print(f"Time per sim (multi, T=1000): {time_per_sim_multi_1000:.4f}s")
print(f"Multi/single ratio: {multi_overhead:.2f}x")

# =========================================================================
# Step 4: T-weighted extrapolation
# =========================================================================
print(f"\n{'='*60}")
print("T-SEGMENTED EXTRAPOLATION (weighted by actual T)")
print(f"{'='*60}")

# Interpolation function based on measured T=1000 and T=10000
# Assume time ≈ a * T^b (power law), fit from two points
# log(t1) = log(a) + b*log(T1), log(t2) = log(a) + b*log(T2)
# b = log(t2/t1) / log(T2/T1)
b = np.log(scaling_ratio) / np.log(10000 / 1000)

def estimate_per_sim_time(T, base_time, T_base=1000, exponent=b):
    """Estimate per-simulation time at T given measured time at T_base."""
    return base_time * (T / T_base) ** exponent

# ---- Single product ----
print(f"\nSingle Product (5T x 2alpha x 500sims x 5algos = {len(DEFAULT_T_VALUES)*len(DEFAULT_ALPHA_VALUES)*500*algo_count_single} sims)")
print(f"  Exponential exponent b = {b:.3f} (from T=1000 vs T=10000)")
print(f"  {'T':>6}  {'sims/config':>12}  {'per-sim':>10}  {'total':>12}")
single_total = 0
for T_val in DEFAULT_T_VALUES:
    n_sims_this_T = len(DEFAULT_ALPHA_VALUES) * 500 * algo_count_single
    per_sim = estimate_per_sim_time(T_val, time_per_sim_1000)
    total_T = n_sims_this_T * per_sim
    single_total += total_T
    print(f"  {T_val:>6}  {n_sims_this_T:>12}  {per_sim:>8.3f}s  {total_T:>8.0f}s ({total_T/60:.1f} min)")

print(f"  {'TOTAL':>6}  {'':>12}  {'':>10}  {single_total:>8.0f}s = {single_total/60:.1f} min = {single_total/3600:.2f} hr")

# ---- Multi product ----
print(f"\nMulti Product (5T x 3demand x 100sims x 3algos = {len(DEFAULT_MULTI_T_VALUES)*3*100*algo_count_multi} sims)")
print(f"  Assume multi/single per-sim ratio of {multi_overhead:.2f}x at all T")
print(f"  {'T':>6}  {'sims/config':>12}  {'per-sim':>10}  {'total':>12}")
multi_total = 0
for T_val in DEFAULT_MULTI_T_VALUES:
    n_sims_this_T = 3 * 100 * algo_count_multi
    per_sim_single = estimate_per_sim_time(T_val, time_per_sim_1000)
    per_sim = per_sim_single * multi_overhead
    total_T = n_sims_this_T * per_sim
    multi_total += total_T
    print(f"  {T_val:>6}  {n_sims_this_T:>12}  {per_sim:>8.3f}s  {total_T:>8.0f}s ({total_T/60:.1f} min)")

print(f"  {'TOTAL':>6}  {'':>12}  {'':>10}  {multi_total:>8.0f}s = {multi_total/60:.1f} min = {multi_total/3600:.2f} hr")

# ---- Extended experiments ----
print(f"\nExtended Experiments (actual config count from main.py):")
# Figure 3: 1 config × 3 algos × 50 sims, T=10000
# Figure 4: 9 alphas × 4 T values = 36 configs × 1 algo × 25 sims, T=500/1000/2000/5000
# Figure 5: 9 alphas × 1 config each × 3 algos × 25 sims, T=1000
# Table 1: 2 configs × 3 algos × 25 sims, T=10000
extended_breakdown = [
    # (desc, T, n_configs, n_algos, n_sims_per)
    ("Fig 3: regret curve", 10000, 1, 3, 50),
    ("Fig 4: heatmap T=500", 500, 9, 1, 25),
    ("Fig 4: heatmap T=1000", 1000, 9, 1, 25),
    ("Fig 4: heatmap T=2000", 2000, 9, 1, 25),
    ("Fig 4: heatmap T=5000", 5000, 9, 1, 25),
    ("Fig 5: delta vs alpha", 1000, 9, 3, 25),
    ("Table 1: final (single)", 10000, 1, 3, 25),
    ("Table 1: final (multi)", 10000, 1, 3, 25),
]

extended_total = 0
print(f"  {'Item':>30}  {'T':>6}  {'per-sim':>10}  {'sims':>6}  {'total':>10}")
for desc, T_val, n_configs, n_algos, n_sims_per in extended_breakdown:
    n_sims_this = n_configs * n_algos * n_sims_per
    # Single product per-sim time unless noted
    per_sim = estimate_per_sim_time(T_val, time_per_sim_1000)
    if desc.endswith('(multi)'):
        per_sim *= multi_overhead
    total_T = n_sims_this * per_sim
    extended_total += total_T
    print(f"  {desc:>30}  {T_val:>6}  {per_sim:>8.3f}s  {n_sims_this:>6}  {total_T:>8.0f}s")

print(f"  {'EXTENDED TOTAL':>30}  {'':>6}  {'':>10}  {'':>6}  {extended_total:>8.0f}s = {extended_total/60:.1f} min = {extended_total/3600:.2f} hr")

# =========================================================================
# Step 5: Summary
# =========================================================================
total_serial = single_total + multi_total + extended_total
cores = 16  # AMD Ryzen 9 7945HX
parallel_speedup = cores * 0.85  # ~85% efficiency from joblib overhead
parallel_time = total_serial / parallel_speedup

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

print(f"\nHardware: AMD Ryzen 9 7945HX ({cores} cores / 32 threads)")
print(f"Assumed parallel efficiency: {parallel_speedup:.0f}x ({cores} × 85%)")
print()

print(f"  {'Experiment':>20}  {'Serial':>15}  {'Parallel':>15}")
print(f"  {'-'*20}  {'-'*15}  {'-'*15}")
print(f"  {'Single Product':>20}  {single_total/3600:>8.2f} hr   {single_total/3600/parallel_speedup*60:>10.0f} min")
print(f"  {'Multi Product':>20}  {multi_total/3600:>8.2f} hr   {multi_total/3600/parallel_speedup*60:>10.0f} min")
print(f"  {'Extended':>20}  {extended_total/3600:>8.2f} hr   {extended_total/3600/parallel_speedup*60:>10.0f} min")
print(f"  {'='*20}  {'='*15}  {'='*15}")
print(f"  {'TOTAL':>20}  {total_serial/3600:>8.2f} hr   {total_serial/3600/parallel_speedup*60:>10.0f} min")

print(f"\n  Quick test (single, n_sim=50): {single_total/(500/50)/3600/parallel_speedup*60:.0f} min parallel")