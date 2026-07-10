from __future__ import annotations
import numpy as np
from .payoffs import well_mixed_expected_payoffs, lattice_realized_average_payoffs


def il_pending_well_mixed(strategies, payoff_matrix, eta, rng):
    pending = strategies.copy()
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)
    if np.isclose(u_c, u_d):
        return pending
    if u_c > u_d:
        eligible = np.where(strategies == 0)[0]
        target = 1
    else:
        eligible = np.where(strategies == 1)[0]
        target = 0

    # AMBIGUITY FROM PAPER:
    # The paper says a ratio eta of the low-payoff agents is randomly selected.
    # Here we implement this as independent Bernoulli thinning.
    # Alternative to test later: sample exactly floor(eta * len(eligible)) agents.
    chosen = eligible[rng.random(len(eligible)) < eta]
    pending[chosen] = target
    return pending


def il_pending_lattice(strategies, payoff_matrix, neighbors, eta, rng):
    pending = strategies.copy()
    payoffs = lattice_realized_average_payoffs(strategies, payoff_matrix, neighbors)
    eligible = []
    targets = {}

    for i, nbrs in enumerate(neighbors):
        my_strategy = strategies[i]
        opposite = [j for j in nbrs if strategies[j] != my_strategy]
        if not opposite:
            continue
        opp_payoffs = np.array([payoffs[j] for j in opposite], dtype=float)
        if payoffs[i] <= np.min(opp_payoffs):
            eligible.append(i)

            # AMBIGUITY FROM PAPER:
            # The text says the agent may adjust after comparing with neighbors of the
            # opposite strategy, but does not fully specify which neighbor is imitated.
            # Here we choose the highest-payoff opposite-strategy neighbor.
            best_j = opposite[int(np.argmax(opp_payoffs))]
            targets[i] = strategies[best_j]

    eligible = np.array(eligible, dtype=int)
    chosen = eligible[rng.random(len(eligible)) < eta] if len(eligible) else eligible
    for i in chosen:
        pending[i] = targets[i]
    return pending
