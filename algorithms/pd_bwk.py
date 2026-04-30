"""
Badanidiyuru et al. (2013) PD-BwK benchmark algorithm.

UCB-based primal-dual approach for bandits with knapsacks.
Simplified implementation: maintains UCB estimates for each price's expected
resource consumption and revenue, then solves LP with these optimistic estimates.
"""

import numpy as np
from typing import Optional
from .base import DynamicPricingAlgorithm
from utils.lp_solver import solve_pricing_lp


class PDBwK(DynamicPricingAlgorithm):
    """
    PD-BwK: UCB-based primal-dual algorithm (Badanidiyuru et al. 2013).

    Maintains UCB estimates of mean demand for each price-product pair.
    At each period, solves LP using UCB demand estimates as an optimistic
    proxy, then samples a price from the LP solution.

    NOTE: This algorithm assumes Bernoulli demand (single-product experiments).
    For Poisson demand, the UCB construction would need to be adjusted.
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)
        # Sufficient statistics for UCB
        self._sum_demand = None    # [K, N] cumulative demand observed
        self._counts = None        # [K] number of times each price chosen
        self._c = None             # fixed capacity rate

    def initialize(self, env):
        self._sum_demand = np.zeros((self.K, self.N))
        self._counts = np.zeros(self.K, dtype=int)
        self._c = np.asarray(env.I, dtype=float) / env.T
        self._initialized = True

    def choose_price(self, t: int, env) -> Optional[int]:
        if not self._initialized:
            raise RuntimeError("Algorithm not initialized")

        # Compute UCB demand estimates (optimistic)
        d_ucb = self._compute_ucb(t)  # [K, N]

        # Solve LP with UCB estimates
        x_opt, _ = solve_pricing_lp(
            prices=self.prices,
            mean_demand=d_ucb,
            consumption_matrix=self.A,
            c=self._c,
        )

        if x_opt is None or not np.any(x_opt > 0):
            return None

        # Sample from LP solution
        close_prob = max(0.0, 1.0 - np.sum(x_opt))
        probs = np.append(x_opt, close_prob)
        choice = self.rng.choice(len(probs), p=probs / probs.sum())

        if choice == len(x_opt):
            return None
        return int(choice)

    def _compute_ucb(self, t: int) -> np.ndarray:
        """
        Compute UCB demand estimates.

        For each (price, product):
          UCB_k = mean_k + sqrt( log(t+1) / (2 * n_k) )
        where mean_k is the empirical average and n_k is the number of observations.
        """
        d_mean = np.zeros((self.K, self.N))
        d_ucb = np.zeros((self.K, self.N))

        for k in range(self.K):
            n_k = max(1, self._counts[k])
            d_mean[k] = self._sum_demand[k] / n_k
            bonus = np.sqrt(np.log(t + 1.0) / (2.0 * n_k))
            d_ucb[k] = d_mean[k] + bonus

        return np.minimum(d_ucb, 1.0)  # cap at 1.0 for Bernoulli

    def update(self, t: int, price_idx: int, demand: np.ndarray):
        if price_idx is not None:
            self._sum_demand[price_idx] += demand
            self._counts[price_idx] += 1