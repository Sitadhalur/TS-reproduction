"""Quick import test to verify all modules load correctly."""
import sys
print(f"Python version: {sys.version}")

from models.demand import BernoulliDemand, PoissonDemand
print("  models.demand: OK")

from models.inventory import InventoryManager
print("  models.inventory: OK")

from models.revenue_network import RevenueNetwork
print("  models.revenue_network: OK")

from models.posterior import BetaPosterior, GammaPosterior, PosteriorFactory
print("  models.posterior: OK")

from algorithms.base import DynamicPricingAlgorithm
print("  algorithms.base: OK")

from algorithms.ts_fixed import TSFixed
print("  algorithms.ts_fixed: OK")

from algorithms.ts_update import TSUpdate
print("  algorithms.ts_update: OK")

from algorithms.ts_unconstrained import TSUnconstrained
print("  algorithms.ts_unconstrained: OK")

from algorithms.bz import BZAlgorithm
print("  algorithms.bz: OK")

from algorithms.pd_bwk import PDBwK
print("  algorithms.pd_bwk: OK")

from utils.lp_solver import solve_pricing_lp
print("  utils.lp_solver: OK")

from utils.statistics import compute_confidence_interval, aggregate_results, cumulative_regret_curve
print("  utils.statistics: OK")

from experiments.config_single import get_single_product_config, DEFAULT_T_VALUES, DEFAULT_ALPHA_VALUES
print("  experiments.config_single: OK")

from experiments.config_multi import get_multi_product_config, DEMAND_FUNCTIONS
print("  experiments.config_multi: OK")

from analysis.baseline import compute_optimal_revenue
print("  analysis.baseline: OK")

from analysis.regret import bayesian_regret_summary
print("  analysis.regret: OK")

from analysis.metrics import compute_metrics_table
print("  analysis.metrics: OK")

print("\n✓ All imports successful!")