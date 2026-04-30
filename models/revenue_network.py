"""
Revenue network environment — simulates the retailer's problem.

Manages the full interaction: demand generation, inventory consumption, revenue calculation.
"""

import numpy as np
from typing import Optional, Callable
from .demand import DemandModel
from .inventory import InventoryManager


class RevenueNetwork:
    """
    Online network revenue management environment.

    At each period t = 1..T:
      - Agent chooses a price vector index k
      - Demand D(t) is realised from F(· ; p_k, θ_true)
      - Revenue = Σ_i p_{ik} * min(D_i(t), inventory-constrained)
      - Inventory is consumed accordingly
    """

    def __init__(
        self,
        N: int,
        M: int,
        K: int,
        prices: np.ndarray,
        consumption_matrix: np.ndarray,
        initial_inventory: np.ndarray,
        T: int,
        demand_model: DemandModel,
        theta_true: np.ndarray,
        seed: Optional[int] = None,
    ):
        """
        Args:
            N: number of products
            M: number of resources
            K: number of price vectors
            prices: [K, N] price matrix
            consumption_matrix: [N, M] resource consumption per product unit
            initial_inventory: [M] initial stock levels
            T: sales horizon length
            demand_model: DemandModel instance for generating demand
            theta_true: true demand parameter (used by demand_model)
            seed: random seed for reproducibility
        """
        self.N = N
        self.M = M
        self.K = K
        self.prices = prices
        self.A = consumption_matrix
        self.I = initial_inventory
        self.T = T
        self.demand_model = demand_model
        self.theta_true = theta_true
        self.rng = np.random.default_rng(seed)

        self.inventory = InventoryManager(initial_inventory)
        self._period = 0

    def reset(self):
        """Reset the environment to initial state."""
        self.inventory.reset()
        self._period = 0

    def step(self, price_idx: int) -> tuple:
        """
        Execute one period with the chosen price.

        Args:
            price_idx: index of chosen price vector (int), or None for "close" price

        Returns:
            demand: [N] realised demand vector
            revenue: float, total revenue for this period
            stock_out: bool, whether any resource ran out
        """
        if price_idx is None:
            # "Closing price": no sales, no consumption
            return np.zeros(self.N), 0.0, False

        # Sample demand from the true model
        demand = self.demand_model.sample(price_idx, self.theta_true, self.rng)

        # Fulfill demand subject to inventory
        actual_sales = self.inventory.consume(demand, self.A)

        # Revenue = price * actual_sales
        revenue = float(np.dot(self.prices[price_idx], actual_sales))

        stock_out = self.inventory.has_stock_out()

        return demand, revenue, stock_out

    def get_inventory_levels(self) -> np.ndarray:
        """Return current inventory [M]."""
        return self.inventory.get_current_levels()

    def get_remaining_capacity_rate(self) -> np.ndarray:
        """
        Return c_j(t) = I_j(t-1) / (T - t + 1) for each resource j.
        Used by TS-update algorithm.
        """
        remaining_stock = self.inventory.get_current_levels()
        remaining_periods = max(1, self.T - self.current_period + 1)
        return remaining_stock / remaining_periods

    @property
    def current_period(self) -> int:
        """Current period (auto-incremented by step())."""
        return self._period
