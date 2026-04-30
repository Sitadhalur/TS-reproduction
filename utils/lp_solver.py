"""
LP solver wrapper for the network revenue management pricing problem.

Solves the linear program from Algorithm 1 (TS-fixed) of
Ferreira, Simchi-Levi & Wang (2015):

    max  Σ_k (Σ_i p_{ik} d_{ik}) x_k
    s.t. Σ_k (Σ_i a_{ij} d_{ik}) x_k ≤ c_j,  ∀j
         Σ_k x_k ≤ 1
         x_k ≥ 0, ∀k

where x_k is the probability of selecting price vector k,
and the "close price" p_∞ is represented by slack (Σ x_k < 1).

Uses scipy.optimize.linprog with HiGHS solver backend.

Shape conventions:
  - prices:       [K, N] — price of product i under price vector k
  - mean_demand:  [K, N] — expected demand of product i under price vector k
  - consumption:  [N, M] — amount of resource j consumed by product i
  - c:            [M]    — per-period resource capacities
"""
import warnings

import numpy as np
from scipy.optimize import linprog


# Suppress the harmless OptimizeWarning about unrecognised 'tol' option
# in scipy's HiGHS wrapper (cosmetic only, does not affect results)
warnings.filterwarnings('ignore', message='Unrecognised options detected.*tol.*')


def solve_pricing_lp(
    prices: np.ndarray,
    mean_demand: np.ndarray,
    consumption_matrix: np.ndarray,
    c: np.ndarray,
    verbose: bool = False,
) -> tuple:
    """
    Solve the pricing LP from Algorithm 1 (TS-fixed).

    The LP decides the probability distribution over price vectors
    that maximises expected per-period revenue subject to inventory
    constraints.

    Args:
        prices: [K, N] array of price vectors.
        mean_demand: [K, N] array where mean_demand[k, i] is the
            expected demand of product i under price vector k.
        consumption_matrix: [N, M] array where A[i, j] is the amount
            of resource j consumed by one unit of product i.
        c: [M] array of per-period resource capacities (c_j = I_j / T).
        verbose: if True, print solver status.

    Returns:
        (x_opt, opt_value) where:
            x_opt: [K] array of optimal selection probabilities.
            opt_value: optimal expected per-period revenue.
    """
    # Validate dimensions
    K, N = prices.shape
    assert mean_demand.shape == (K, N), (
        f"mean_demand shape {mean_demand.shape} != ({K}, {N})"
    )
    M = consumption_matrix.shape[1]
    assert c.shape == (M,), f"c shape {c.shape} != ({M},)"

    # Step 1: Compute objective coefficients
    # r_k = Σ_i p_{ik} * d_{ik}
    # Both prices and mean_demand are [K, N], so element-wise product sums over N
    r = np.sum(prices * mean_demand, axis=1)  # [K]

    # Step 2: Compute constraint coefficients
    # For each resource j: Σ_k (Σ_i a_{ij} d_{ik}) x_k ≤ c_j
    # resource_usage[j, k] = Σ_i A[i, j] * mean_demand[k, i]
    resource_usage = np.zeros((M, K))
    for k in range(K):
        for j in range(M):
            resource_usage[j, k] = np.sum(
                consumption_matrix[:, j] * mean_demand[k, :]
            )

    # Step 3: Build LP in standard form for linprog
    # linprog solves: min c_obj^T x, subject to A_ub x ≤ b_ub, x ≥ 0
    #
    # Our LP:
    #   max r^T x
    #   s.t. resource_usage @ x ≤ c
    #        sum(x) ≤ 1
    #        x ≥ 0
    #
    # Convert to min: min -r^T x

    c_obj = -r  # negate because linprog minimises

    # Inequality constraints: A_ub @ x ≤ b_ub
    # Rows 0..M-1: resource constraints
    # Row M: sum(x) ≤ 1
    A_ub = np.vstack([resource_usage, np.ones(K)])
    b_ub = np.hstack([c, np.array([1.0])])

    # Variable bounds: x_k ≥ 0
    bounds = [(0, None)] * K

    # Step 4: Solve
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        result = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
            options={'tol': 1e-8, 'maxiter': 5000},
        )

    if not result.success:
        if verbose:
            print(f"LP solver warning: {result.message}. "
                  f"Falling back to equal-distribution.")
        # Fallback: uniform distribution over price vectors
        x_opt = np.ones(K) / K
        opt_value = np.dot(r, x_opt)
    else:
        x_opt = result.x
        opt_value = -result.fun  # negate back to maximisation

    return x_opt, opt_value


def solve_pricing_lp_with_close_price(
    prices: np.ndarray,
    mean_demand: np.ndarray,
    consumption_matrix: np.ndarray,
    c: np.ndarray,
    verbose: bool = False,
) -> tuple:
    """
    Solve pricing LP and return a full probability vector including
    the "close price" option (index K, representing not selling).

    The close price has revenue 0 and consumes 0 resources.

    Returns:
        (x_full, opt_value) where:
            x_full: [K+1] array, x_full[K] is the probability of
                    selecting the close price (i.e. 1 - sum(x_opt)).
            opt_value: optimal expected per-period revenue.
    """
    x_opt, opt_value = solve_pricing_lp(
        prices, mean_demand, consumption_matrix, c,
        verbose=verbose,
    )

    # Close price probability = 1 - Σ x_k
    close_prob = max(0.0, 1.0 - np.sum(x_opt))

    # Normalise if needed (numerical tolerance)
    total = np.sum(x_opt) + close_prob
    if total > 1.0 + 1e-10:
        x_opt = x_opt / total
        close_prob = close_prob / total

    x_full = np.append(x_opt, close_prob)
    return x_full, opt_value