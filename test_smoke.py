"""
Quick smoke test: run a tiny single-product experiment to verify end-to-end functionality.
"""
import sys
import os
import numpy as np

# Minimal test config
N, M, K = 1, 1, 4
prices = np.array([[29.90], [34.90], [39.90], [44.90]])
theta_true = np.array([[0.8], [0.6], [0.3], [0.1]])  # true mean demands
A = np.ones((N, M))  # consumption matrix
I = np.array([250.0])
T = 100

# Test LP solver
print("Testing LP solver...")
from utils.lp_solver import solve_pricing_lp
c = I / T
x_opt = solve_pricing_lp(prices, theta_true, A, c)
print(f"  LP solution: x = {np.round(x_opt, 4)}")
print(f"  Sum of x = {np.sum(x_opt):.4f}")

# Test baseline
print("Testing baseline...")
config_dict = {
    'prices': prices, 'theta_true': theta_true, 'A': A, 'I': I, 'T': T,
    'N': N, 'M': M, 'K': K,
}
from analysis.baseline import compute_optimal_revenue
baseline = compute_optimal_revenue(config_dict)
print(f"  Optimal total revenue: {baseline:.4f}")

# Test demand model
print("Testing demand model...")
from models.demand import BernoulliDemand
demand_model = BernoulliDemand(theta_true)  # mean_demand = theta_true
rng = np.random.default_rng(42)
sample = demand_model.sample(0, theta_true, rng)
print(f"  Demand sample: {sample}")

# Test inventory
print("Testing inventory...")
from models.inventory import InventoryManager
inv = InventoryManager(I)
print(f"  Initial inventory: {inv.get_current_levels()}")
inv.consume(np.array([0.0]), A)
print(f"  After no consumption: {inv.get_current_levels()}")
inv.reset()
inv.consume(np.array([1.0]), A)
print(f"  After consuming 1: {inv.get_current_levels()}")

# Test revenue network
print("Testing RevenueNetwork...")
from models.revenue_network import RevenueNetwork
env = RevenueNetwork(N, M, K, prices, A, I, T, demand_model, theta_true, seed=42)
env.reset()
print(f"  Environment created OK")

# Test posterior
print("Testing posterior...")
from models.posterior import PosteriorFactory
post = PosteriorFactory.create('beta', K=K, N=N)
print(f"  Posterior type: {post.__class__.__name__}")
post_rng = np.random.default_rng(99)
s_full = post.sample(post_rng)
print(f"    Posterior sample (all prices): {np.round(s_full, 4)}")
print(f"    Posterior mean: {np.round(post.get_mean(), 4)}")

# Test TS-fixed
print("Testing TS-fixed...")
from algorithms.ts_fixed import TSFixed
algo = TSFixed(config_dict)
algo.initialize(env)
print(f"  Algorithm initialized with c = {algo.c}")
price_idx = algo.choose_price(1, env)
print(f"  Chosen price index: {price_idx}")
algo.update(1, price_idx, np.array([0.0]))
print(f"  Updated after observing 0 demand")

# Test TS-update
print("Testing TS-update...")
from algorithms.ts_update import TSUpdate
algo2 = TSUpdate(config_dict)
algo2.initialize(env)
price_idx2 = algo2.choose_price(1, env)
print(f"  Chosen price index: {price_idx2}")
algo2.update(1, price_idx2, np.array([1.0]))
print(f"  Updated after observing 1 demand")

# Test BZ
print("Testing BZ...")
from algorithms.bz import BZAlgorithm
algo3 = BZAlgorithm(config_dict)
algo3.initialize(env)
price_idx3 = algo3.choose_price(1, env)
print(f"  BZ chosen price index: {price_idx3}")

# Test PD-BwK
print("Testing PD-BwK...")
from algorithms.pd_bwk import PDBwK
algo4 = PDBwK(config_dict)
algo4.initialize(env)
price_idx4 = algo4.choose_price(1, env)
print(f"  PD-BwK chosen price index: {price_idx4}")

# Test TS-unconstrained
print("Testing TS-unconstrained...")
from algorithms.ts_unconstrained import TSUnconstrained
algo5 = TSUnconstrained(config_dict)
algo5.initialize(env)
price_idx5 = algo5.choose_price(1, env)
print(f"  TS-unconstrained chosen price index: {price_idx5}")

# Test full simulation loop
print("\nRunning full simulation loop (T=20)...")
from experiments.runner import run_single_simulation
for name, cls in [('TS-fixed', TSFixed), ('TS-update', TSUpdate), ('BZ', BZAlgorithm), ('PD-BwK', PDBwK), ('TS-unconstrained', TSUnconstrained)]:
    algo_instance = cls(config_dict)
    rev, per_period = run_single_simulation(config_dict, algo_instance, env_seed=42, algo_seed=123)
    pct = (rev / baseline) * 100
    print(f"  {name}: revenue={rev:.2f}, %optimal={pct:.1f}%")

print("\n✓ All smoke tests passed!")