"""
Posterior distribution management for Thompson Sampling.

Provides:
- BetaPosterior: for Bernoulli demand (single-product)
- GammaPosterior: for Poisson demand (multi-product)
- PosteriorFactory: convenience factory
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict


class PosteriorManager(ABC):
    """Abstract base for posterior distribution tracking."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample the demand parameter from the current posterior.
        Returns shape matching the expected theta.
        """
        ...

    @abstractmethod
    def update(self, price_idx: int, demand: np.ndarray):
        """
        Update posterior given observed demand at chosen price.
        """
        ...

    @abstractmethod
    def get_mean(self) -> np.ndarray:
        """Return the posterior mean estimate of the demand parameter."""
        ...


class BetaPosterior(PosteriorManager):
    """
    Beta-Bernoulli conjugate posterior.
    For each (price_idx, product) pair: Beta(alpha_k, beta_k).
    """

    def __init__(self, K: int, N: int, alpha_init: float = 1.0, beta_init: float = 1.0):
        """
        Args:
            K: number of price vectors
            N: number of products
            alpha_init, beta_init: prior Beta(alpha_init, beta_init) parameters
        """
        self.K = K
        self.N = N
        self.alphas = np.full((K, N), alpha_init)   # [K, N]
        self.betas = np.full((K, N), beta_init)      # [K, N]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample mean demand [K, N] from current Beta posteriors."""
        return rng.beta(self.alphas, self.betas)  # [K, N]

    def update(self, price_idx: int, demand: np.ndarray):
        """
        Update Beta posterior: add successes (demand_i) and failures (1 - demand_i).
        Assumes Bernoulli demand (0 or 1).
        """
        self.alphas[price_idx] += demand           # add purchases
        self.betas[price_idx] += (1.0 - demand)    # add no-purchases

    def get_mean(self) -> np.ndarray:
        """Return posterior mean: alpha / (alpha + beta) for each [K, N]."""
        denom = self.alphas + self.betas
        return np.where(denom > 0, self.alphas / denom, 0.5)


class GammaPosterior(PosteriorManager):
    """
    Gamma-Poisson conjugate posterior.
    For each (price_idx, product) pair: Gamma(shape_k, rate_k).
    Prior: Gamma(1, 1) (exponential with mean 1).
    """

    def __init__(self, K: int, N: int, shape_init: float = 1.0, rate_init: float = 1.0):
        """
        Args:
            K: number of price vectors
            N: number of products
            shape_init, rate_init: prior Gamma(shape_init, rate_init) parameters
        """
        self.K = K
        self.N = N
        self.shapes = np.full((K, N), shape_init)   # [K, N] — alpha parameter
        self.rates = np.full((K, N), rate_init)      # [K, N] — beta (rate) parameter
        self.counts = np.zeros(K, dtype=int)         # [K] — times each price chosen

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample mean demand [K, N] from current Gamma posteriors."""
        return rng.gamma(self.shapes, 1.0 / self.rates)  # [K, N]

    def update(self, price_idx: int, demand: np.ndarray):
        """
        Update Gamma posterior:
          shape_k += demand_i   (cumulative demand)
          rate_k  += 1          (each observation adds 1 to rate)
        """
        self.shapes[price_idx] += demand           # W_ik += demand_i
        self.rates[price_idx] += 1.0               # N_k += 1
        self.counts[price_idx] += 1

    def get_mean(self) -> np.ndarray:
        """Return posterior mean: shape / rate for each [K, N]."""
        return np.where(self.rates > 0, self.shapes / self.rates, 1.0)


class PosteriorFactory:
    """Factory to create appropriate posterior manager based on demand type."""

    @staticmethod
    def create(prior_type: str, K: int, N: int, **kwargs) -> PosteriorManager:
        """
        Args:
            prior_type: 'beta' or 'gamma'
            K: number of price vectors
            N: number of products

        Returns:
            PosteriorManager instance
        """
        if prior_type == 'beta':
            return BetaPosterior(K, N,
                                 alpha_init=kwargs.get('alpha_init', 1.0),
                                 beta_init=kwargs.get('beta_init', 1.0))
        elif prior_type == 'gamma':
            return GammaPosterior(K, N,
                                  shape_init=kwargs.get('shape_init', 1.0),
                                  rate_init=kwargs.get('rate_init', 1.0))
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")