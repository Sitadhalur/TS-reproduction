"""
Classic Thompson Sampling without inventory constraints (baseline).

This does NOT use the LP sub-routine. It simply samples from the posterior
mean for each price and picks the price with the highest expected revenue.
We expect this to perform poorly because the optimal strategy under inventory
constraints is a *mix* of prices, not a single price.
"""

import numpy as np
from typing import Optional
from .base import DynamicPricingAlgorithm
from models.posterior import PosteriorFactory


class TSUnconstrained(DynamicPricingAlgorithm):
    """
    Classic Thompson Sampling baseline (no inventory constraints).

    At each period t:
      1. Sample mean demand for each price from posterior
      2. Compute expected revenue = p_k · μ_k
      3. Pick the price with the highest expected revenue (greedy w.r.t. sample)
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.prior_type = config.get('prior_type', 'beta')

    def initialize(self, env):
        self.posterior = PosteriorFactory.create(self.prior_type, self.K, self.N)
        self._initialized = True

    def choose_price(self, t: int, env) -> Optional[int]:
        # Sample mean demand from posterior
        d_sample = self.posterior.sample(self.rng)  # [K, N]

        # Compute expected revenue for each price: r_k = Σ_i p_{ki} * d_{ki}
        revenues = np.sum(self.prices * d_sample, axis=1)  # [K]

        # Pick the price with highest sampled revenue (greedy)
        best_k = int(np.argmax(revenues))
        return best_k

    def update(self, t: int, price_idx: int, demand: np.ndarray):
        if price_idx is not None:
            self.posterior.update(price_idx, demand)