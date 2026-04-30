"""
Algorithm 5: TS-BwK — Thompson Sampling for Bandits with Knapsacks.

Generalisation of TS-fixed that stops once any resource is exhausted.
Similar to TS-fixed but the algorithm halts if stock-out occurs.
"""

import numpy as np
from typing import Optional
from .ts_fixed import TSFixed
from utils.lp_solver import solve_pricing_lp


class TSBwK(TSFixed):
    """
    TS-BwK: Thompson Sampling for Bandits with Knapsacks.

    Same as TS-fixed, but if any resource is exhausted, the algorithm stops
    making further sales (closes).
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)
        self._stopped = False

    def initialize(self, env):
        super().initialize(env)
        self._stopped = False

    def choose_price(self, t: int, env) -> Optional[int]:
        if self._stopped:
            return None  # permanently closed

        # Check if any resource is exhausted
        if env.has_stock_out():
            self._stopped = True
            return None

        return super().choose_price(t, env)