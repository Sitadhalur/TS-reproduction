"""
Demand distribution models.

Provides abstract base and concrete implementations for:
- Bernoulli demand (single-product, buy / no-buy)
- Poisson demand (multi-product, integer arrivals)
- Linear, Exponential, Logit mean-demand functions (multi-product)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class DemandModel(ABC):
    """Abstract base class for demand generation."""

    @abstractmethod
    def sample(self, price_idx: int, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Return demand vector [N] given the chosen price index and true parameter theta.

        Args:
            price_idx: index of the chosen price vector
            theta: true demand parameter(s)
            rng: numpy random generator for reproducibility

        Returns:
            demand: array of shape (N,) with realised demands for each product
        """
        ...

    @abstractmethod
    def mean(self, price_idx: int, theta: np.ndarray) -> np.ndarray:
        """
        Return expected demand vector [N] under parameter theta.
        """
        ...


class BernoulliDemand(DemandModel):
    """
    Bernoulli demand model: each customer either buys (1) or not (0).
    Used in single-product experiments (Section 3.1).
    """

    def __init__(self, mean_demand: np.ndarray):
        """
        Args:
            mean_demand: [K, N] array of purchase probabilities for each price-product pair
        """
        self.mean_demand = mean_demand

    def sample(self, price_idx: int, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        prob = self.mean_demand[price_idx]  # [N]
        return (rng.uniform(size=prob.shape) < prob).astype(float)

    def mean(self, price_idx: int, theta: np.ndarray) -> np.ndarray:
        return self.mean_demand[price_idx]


class PoissonDemand(DemandModel):
    """
    Poisson demand model: each product has Poisson(lambda) arrivals.
    lambda is determined by the price vector and a mean-demand function.
    Used in multi-product experiments (Section 3.2).
    """

    def __init__(self, mean_function: str = 'linear', prices: np.ndarray = None):
        """
        Args:
            mean_function: one of 'linear', 'exponential', 'logit'
            prices: [K, N] price matrix, needed to compute mean demands
        """
        self.mean_function = mean_function
        self.prices = prices

    def _compute_rate(self, price_idx: int, theta: np.ndarray) -> np.ndarray:
        """
        Compute Poisson rate for each product given price index and theta.

        theta encoding for each mean function:
          linear:       theta = [intercept1, slope1, intercept2, slope2]  (len=2N)
          exponential:  theta = [scale1, rate1, scale2, rate2]           (len=2N)
          logit:        theta = []  (no parameters, deterministic function of price)

        NOTE: For the multi-product experiments in the paper, theta_true is simply
        the pre-computed mean demand [K, N] for each price-product pair.
        """
        if self.prices is None:
            raise ValueError("PoissonDemand requires prices to be set")

        p = self.prices[price_idx]  # [N]

        if self.mean_function == 'linear':
            intercepts = theta[0::2]
            slopes = theta[1::2]
            return np.maximum(intercepts - slopes * p, 0.0)
        elif self.mean_function == 'exponential':
            scales = theta[0::2]
            rates = theta[1::2]
            return scales * np.exp(-rates * p)
        elif self.mean_function == 'logit':
            denom = 1.0 + np.sum(np.exp(-p))
            return 10.0 * np.exp(-p) / denom
        else:
            raise ValueError(f"Unknown mean function: {self.mean_function}")

    def _direct_sample(self, price_idx: int, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Simple sampling when theta_true is a [K, N] matrix of pre-computed means.
        """
        lam = theta[price_idx]  # [N] — pre-computed mean demand
        return rng.poisson(lam).astype(float)

    def sample(self, price_idx: int, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # If theta is a 2D array matching [K, N], use direct lookup
        if theta.ndim == 2 and theta.shape[0] > 2:
            return self._direct_sample(price_idx, theta, rng)
        lam = self._compute_rate(price_idx, theta)
        return rng.poisson(lam).astype(float)

    def mean(self, price_idx: int, theta: np.ndarray) -> np.ndarray:
        if theta.ndim == 2 and theta.shape[0] > 2:
            return theta[price_idx]
        return self._compute_rate(price_idx, theta)


# --- Convenience aliases for multi-product mean-demand configurations ---

class LinearDemand(PoissonDemand):
    def __init__(self):
        super().__init__(mean_function='linear')


class ExponentialDemand(PoissonDemand):
    def __init__(self):
        super().__init__(mean_function='exponential')


class LogitDemand(PoissonDemand):
    def __init__(self):
        super().__init__(mean_function='logit')