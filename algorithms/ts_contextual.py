"""
Algorithm 4: TS-contextual — Thompson Sampling with Contextual Information.

Extension that incorporates customer context (e.g., member vs. non-member).
Maintains separate posteriors for each context class.
"""

import numpy as np
from typing import Optional, Dict
from .base import DynamicPricingAlgorithm
from models.posterior import PosteriorFactory


class TSContextual(DynamicPricingAlgorithm):
    """
    TS-contextual: handles X customer contexts with different demand elasticities.

    Maintains separate posterior for each context x in X.
    At each period, observes context first, then applies TS using context-specific posterior.
    """

    def __init__(self, config: dict, seed: Optional[int] = None):
        super().__init__(config, seed)
        self.num_contexts = config.get('num_contexts', 2)
        self.prior_type = config.get('prior_type', 'beta')
        self._posteriors = {}  # context_idx -> PosteriorManager

    def initialize(self, env):
        """Create separate posteriors for each context."""
        self._posteriors = {
            x: PosteriorFactory.create(self.prior_type, self.K, self.N)
            for x in range(self.num_contexts)
        }
        self._initialized = True

    def choose_price(self, t: int, env) -> Optional[int]:
        # Simplified: sample from context-averaged posterior
        # Full implementation would observe context first
        d_avg = np.zeros((self.K, self.N))
        for x in range(self.num_contexts):
            d_avg += self._posteriors[x].sample(self.rng)
        d_avg /= self.num_contexts

        # Best price by expected revenue
        revenues = np.sum(self.prices * d_avg, axis=1)
        return int(np.argmax(revenues))

    def update(self, t: int, price_idx: int, demand: np.ndarray):
        if price_idx is not None:
            # Simplified: update all contexts equally
            # Full implementation would update based on observed context
            for x in range(self.num_contexts):
                self._posteriors[x].update(price_idx, demand)