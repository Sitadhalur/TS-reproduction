"""
Algorithm 2: Thompson Sampling with Dynamic Inventory Constraints (TS-update).

Same as TS-fixed, but replaces fixed c_j with time-varying c_j(t) = I_j(t-1) / (T - t + 1).
"""

import numpy as np
from typing import Optional
from .ts_fixed import TSFixed
from utils.lp_solver import solve_pricing_lp


class TSUpdate(TSFixed):
    """
    TS-update: Thompson Sampling with dynamic (re-solving) inventory constraints.

    Differs from TS-fixed only in how c_j is computed:
      - TS-fixed:  c_j = I_j / T
      - TS-update: c_j(t) = I_j(t-1) / (T - t + 1)
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)

    def choose_price(self, t: int, env) -> Optional[int]:
        """
        Sample -> LP with time-varying capacity -> sample price.

        Uses env.get_inventory_levels() for c_j(t).
        """
        # Step 1: Sample mean demand from posterior
        d_sample = self.posterior.sample(self.rng)  # [K, N]

        # Step 2: Get updated capacity rates from environment
        remaining_stock = env.get_inventory_levels()
        remaining_time = max(1, self.T - t + 1)
        c_t = np.asarray(remaining_stock, dtype=float) / remaining_time

        # Solve LP with time-varying constraints
        x_opt, _ = solve_pricing_lp(
            prices=self.prices,
            mean_demand=d_sample,
            consumption_matrix=self.A,
            c=c_t,
        )

        if x_opt is None or not np.any(x_opt > 0):
            return None

        # Step 3: Sample price from optimal distribution
        # Clip tiny negative values from numerical error in LP solver
        x_opt = np.clip(x_opt, 0.0, None)
        close_prob = max(0.0, 1.0 - np.sum(x_opt))
        probs = np.append(x_opt, close_prob)
        choice = self.rng.choice(len(probs), p=probs / probs.sum())

        if choice == len(x_opt):
            return None
        return int(choice)