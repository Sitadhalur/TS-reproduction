"""
Inventory management for the revenue network environment.

Tracks resource consumption and detects stock-outs.
"""

import numpy as np


class InventoryManager:
    """
    Manages inventory levels for M resources across a sales horizon.

    Supports two stock-out handling modes (Section 2 of the paper):
      (a) Full satisfaction: all demand is met if inventory allows.
      (b) Partial satisfaction: excess demand is lost once inventory runs out.
    """

    def __init__(self, initial_inventory: np.ndarray):
        """
        Args:
            initial_inventory: [M] array of initial stock levels
        """
        self.initial_inventory = initial_inventory.copy()
        self.reset()

    def reset(self):
        """Reset inventory to initial levels."""
        self.current_inventory = self.initial_inventory.copy()
        self.stock_out_periods = [False] * len(self.initial_inventory)

    def get_current_levels(self) -> np.ndarray:
        """Return current inventory [M]."""
        return self.current_inventory.copy()

    def has_stock_out(self) -> bool:
        """Return True if any resource is exhausted."""
        return np.any(self.current_inventory <= 0)

    def consume(self, demand: np.ndarray, consumption_matrix: np.ndarray) -> np.ndarray:
        """
        Attempt to satisfy demand given current inventory.

        Args:
            demand: [N] vector of realised demand for each product
            consumption_matrix: [N, M] matrix a_{ij} = units of resource j per product i

        Returns:
            actual_sales: [N] vector of actually fulfilled demand (may be less than demand
                          if inventory insufficient)
        """
        # Total resource requirement: demand_i * a_{ij} summed over i
        required = np.dot(demand, consumption_matrix)  # [M]

        # Check if any resource would go negative
        if np.all(required <= self.current_inventory + 1e-12):
            # Case (a): fully satisfy demand
            actual_sales = demand.copy()
            self.current_inventory -= required
        else:
            # Case (b): partially satisfy — scale down demand proportionally
            # along each resource dimension. Take the most constrained ratio.
            # This is a simple proportional rationing scheme.
            ratios = np.full_like(required, np.inf, dtype=float)
            positive_required = required > 1e-12
            ratios[positive_required] = (
                self.current_inventory[positive_required]
                / required[positive_required]
            )
            scale = np.min(ratios)
            scale = max(scale, 0.0)  # ensure non-negative
            actual_sales = demand * scale
            self.current_inventory -= np.dot(actual_sales, consumption_matrix)
            self.current_inventory = np.maximum(self.current_inventory, 0.0)

        # Mark stock-out for depleted resources
        for j in range(len(self.current_inventory)):
            if self.current_inventory[j] <= 0:
                self.stock_out_periods[j] = True

        return actual_sales

    def remaining_fraction(self) -> np.ndarray:
        """Return fraction of initial inventory still available [M]."""
        return self.current_inventory / np.maximum(self.initial_inventory, 1e-12)