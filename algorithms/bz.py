"""
Besbes & Zeevi (2012) benchmark algorithm.

Explore-then-exploit strategy:
  - First tau = T^{2/3} periods: explore each price equally
  - Remaining periods: exploit the best price based on exploration data
"""

import numpy as np
from typing import Optional
from .base import DynamicPricingAlgorithm
from models.posterior import PosteriorFactory
from utils.lp_solver import solve_pricing_lp


class BZAlgorithm(DynamicPricingAlgorithm):
    """
    Besbes & Zeevi (2012) explore-then-exploit.

    Exploration phase: cycles through all K prices equally.
    Exploitation phase: uses estimated demand to solve LP once and
    then uses the optimal mixing distribution for the remaining periods.
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.prior_type = config.get('prior_type', 'beta')
        self.tau = None  # exploration horizon
        self._exploit_policy = None  # cached LP solution for exploitation
        self._exploration_counts = None  # [K] how many times each price was tried
        self._exploration_demand_sums = None  # [K, N] cumulative observed demand

    def initialize(self, env):
        """Set exploration horizon tau = T^{2/3}."""
        self.tau = int(np.ceil(self.T ** (2.0 / 3.0)))
        self._exploit_policy = None
        self._exploration_counts = np.zeros(self.K, dtype=int)
        self._exploration_demand_sums = np.zeros((self.K, self.N), dtype=float)
        self.posterior = PosteriorFactory.create(self.prior_type, self.K, self.N)
        self._initialized = True

    def _get_exploration_mean_demand(self) -> np.ndarray:
        """
        Return the empirical exploration mean demand matrix [K, N].

        BZ is an explore-then-exploit benchmark, so the exploitation LP should
        be driven by the empirical demand estimates collected during the
        exploration phase rather than by a Thompson-sampling posterior mean.
        """
        d_hat = np.zeros((self.K, self.N), dtype=float)
        explored = self._exploration_counts > 0
        if np.any(explored):
            d_hat[explored] = (
                self._exploration_demand_sums[explored]
                / self._exploration_counts[explored, None]
            )

        # Safety fallback: if a price was never explored for any reason, use
        # the posterior mean for that arm only.
        if not np.all(explored):
            posterior_mean = self.posterior.get_mean()
            d_hat[~explored] = posterior_mean[~explored]

        return d_hat

    def choose_price(self, t: int, env) -> Optional[int]:
        if t <= self.tau:
            # Exploration phase: cycle through prices
            return (t - 1) % self.K
        else:
            # Exploitation phase: use the best LP-determined mix
            if self._exploit_policy is None:
                self._compute_exploit_policy(env)
            if self._exploit_policy is None:
                return 0  # fallback
            # Sample from the fixed exploit distribution
            probs = np.append(self._exploit_policy,
                              max(0.0, 1.0 - np.sum(self._exploit_policy)))
            choice = self.rng.choice(len(probs), p=probs / probs.sum())
            if choice == len(self._exploit_policy):
                return None
            return int(choice)

    def _compute_exploit_policy(self, env):
        """Compute exploitation LP from empirical demand estimates [K, N]."""
        d_mean = self._get_exploration_mean_demand()  # [K, N]
        remaining_time = self.T - self.tau
        if remaining_time <= 0:
            remaining_time = 1
        # Use actual remaining inventory (not an estimate)
        remaining_stock = env.get_inventory_levels()
        c = remaining_stock / remaining_time

        x_opt, _ = solve_pricing_lp(
            prices=self.prices,
            mean_demand=d_mean,
            consumption_matrix=self.A,
            c=c,
        )
        self._exploit_policy = x_opt

    def update(self, t: int, price_idx: int, demand: np.ndarray):
        """Update posterior during exploration phase only."""
        if t <= self.tau and price_idx is not None:
            self._exploration_counts[price_idx] += 1
            self._exploration_demand_sums[price_idx] += demand
            self.posterior.update(price_idx, demand)