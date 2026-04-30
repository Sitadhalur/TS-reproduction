"""
Abstract base class for all dynamic pricing algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class DynamicPricingAlgorithm(ABC):
    """
    All pricing algorithms inherit from this base class.

    Lifecycle:
        1. initialize(env)          — set up internal state
        2. choose_price(t, env)     — return price_idx for period t
        3. update(t, price_idx, demand) — incorporate new observation
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        """
        Args:
            config: dictionary containing at least N, M, K, prices, A, I, T, etc.
            seed: random seed for algorithm internals (sampling, exploration, etc.)
        """
        self.config = config
        self.N = config['N']
        self.M = config['M']
        self.K = config['K']
        self.prices = config['prices']          # [K, N]
        self.A = config['A']                    # [N, M]
        self.I = np.asarray(config['I']).flatten() if 'I' in config else None
        self.T = config['T']

        self.rng = np.random.default_rng(seed)
        self._initialized = False

    @abstractmethod
    def initialize(self, env):
        """
        Initialise algorithm state given the environment.

        Args:
            env: RevenueNetwork instance (provides access to N, M, K, T, etc.)
        """
        ...

    @abstractmethod
    def choose_price(self, t: int, env) -> int:
        """
        Choose a price vector for period t.

        Args:
            t: current period (1-indexed)
            env: RevenueNetwork instance (provides inventory state for TS-update)

        Returns:
            price_idx: int in [0, K-1], or None for "close" price
        """
        ...

    @abstractmethod
    def update(self, t: int, price_idx: int, demand: np.ndarray):
        """
        Update internal state after observing demand.

        Args:
            t: current period (1-indexed)
            price_idx: the price that was chosen
            demand: [N] observed demand vector
        """
        ...