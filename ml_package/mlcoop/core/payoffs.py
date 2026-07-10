from __future__ import annotations
import numpy as np


def well_mixed_expected_payoffs(strategies: np.ndarray, payoff_matrix: np.ndarray):
    N = len(strategies)
    x_c = int(np.sum(strategies))
    x_d = N - x_c
    a11, a12 = payoff_matrix[0, 0], payoff_matrix[0, 1]
    a21, a22 = payoff_matrix[1, 0], payoff_matrix[1, 1]
    u_c = (a11 * max(x_c - 1, 0) + a12 * x_d) / (N - 1)
    u_d = (a21 * x_c + a22 * max(x_d - 1, 0)) / (N - 1)
    return float(u_c), float(u_d)


def well_mixed_counterfactual_payoffs(strategies: np.ndarray, payoff_matrix: np.ndarray):
    # AMBIGUITY FROM PAPER:
    # The extracted text says RL uses “pure payoffs” derived from the payoff matrix
    # and opponents, but does not operationalize this completely for the well-mixed case.
    # Here we use population-level expected payoffs for choosing C and D.
    # Alternative to test later: sample one or more opponents per agent per round.
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)
    N = len(strategies)
    return np.full(N, u_c), np.full(N, u_d)


def lattice_realized_average_payoffs(strategies: np.ndarray, payoff_matrix: np.ndarray, neighbors):
    payoffs = np.zeros(len(strategies), dtype=float)
    for i, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        s_i = strategies[i]
        total = 0.0
        for j in nbrs:
            s_j = strategies[j]
            if s_i == 1 and s_j == 1:
                total += payoff_matrix[0, 0]
            elif s_i == 1 and s_j == 0:
                total += payoff_matrix[0, 1]
            elif s_i == 0 and s_j == 1:
                total += payoff_matrix[1, 0]
            else:
                total += payoff_matrix[1, 1]
        payoffs[i] = total / len(nbrs)
    return payoffs


def lattice_counterfactual_payoffs(strategies: np.ndarray, payoff_matrix: np.ndarray, neighbors):
    payoff_if_c = np.zeros(len(strategies), dtype=float)
    payoff_if_d = np.zeros(len(strategies), dtype=float)
    for i, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        total_c = 0.0
        total_d = 0.0
        for j in nbrs:
            s_j = strategies[j]
            if s_j == 1:
                total_c += payoff_matrix[0, 0]
                total_d += payoff_matrix[1, 0]
            else:
                total_c += payoff_matrix[0, 1]
                total_d += payoff_matrix[1, 1]
        payoff_if_c[i] = total_c / len(nbrs)
        payoff_if_d[i] = total_d / len(nbrs)
    return payoff_if_c, payoff_if_d
