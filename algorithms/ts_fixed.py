"""
Algorithm 1: Thompson Sampling with Fixed Inventory Constraints (TS-fixed).

Implements the core TS + LP algorithm described in Section 3.
"""

import numpy as np
from typing import Optional
from .base import DynamicPricingAlgorithm
from models.posterior import PosteriorFactory
from utils.lp_solver import solve_pricing_lp


class TSFixed(DynamicPricingAlgorithm):
    """
    TS-fixed: Thompson Sampling with fixed inventory constraints.

    At each period t:
      1. Sample θ(t) from posterior P(θ | H_{t-1})
      2. Compute mean demand d(t) = E[D | θ(t)]
      3. Solve LP(d(t)) to get mixed strategy x(t)
      4. Sample price index from x(t)
      5. Observe demand and update posterior
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.prior_type = config.get('prior_type', 'beta')
        self.demand_type = config.get('demand_type', 'bernoulli')
        self.c = None  # fixed capacity rate c_j = I_j / T

    def initialize(self, env):
        """Initialise posterior and fixed capacity rates."""
        self.posterior = PosteriorFactory.create(self.prior_type, self.K, self.N)
        # Fixed capacity: c_j = I_j / T
        self.c = np.asarray(env.I, dtype=float) / env.T
        self._initialized = True

    def choose_price(self, t: int, env) -> Optional[int]:
        """
        Thompson Sampling step: sample -> LP -> sample price.

        Returns:
            price_idx: int in [0, K-1] or None (close price)
        """
        # Step 1: Sample mean demand from posterior
        d_sample = self.posterior.sample(self.rng)  # [K, N]

        # Step 2: Solve LP to get mixing distribution
        x_opt, _ = solve_pricing_lp(
            prices=self.prices,
            mean_demand=d_sample,
            consumption_matrix=self.A,
            c=self.c,
        )

        if x_opt is None or not np.any(x_opt > 0):
            # No feasible solution — choose "close" price
            return None

        # Step 3: Sample price from the optimal mixing distribution
        # Clip tiny negative values from numerical error in LP solver
        x_opt = np.clip(x_opt, 0.0, None)
        close_prob = max(0.0, 1.0 - np.sum(x_opt))
        probs = np.append(x_opt, close_prob)
        choice = self.rng.choice(len(probs), p=probs / probs.sum())

        if choice == len(x_opt):
            return None  # close price
        return int(choice)

    def update(self, t: int, price_idx: int, demand: np.ndarray):
        """Update posterior with observed demand."""
        if price_idx is not None:
            self.posterior.update(price_idx, demand)