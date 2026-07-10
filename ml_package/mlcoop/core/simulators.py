from __future__ import annotations
from dataclasses import asdict
import numpy as np
from .games import get_payoff_matrix
from .utils import make_rng, initialize_strategies, fraction_cooperators
from .topology import build_square_lattice_graph, neighbor_lists_from_graph
from .payoffs import well_mixed_expected_payoffs, well_mixed_counterfactual_payoffs, lattice_counterfactual_payoffs
from .il import il_pending_well_mixed, il_pending_lattice
from .rl import ReinforcementState, rl_pending_strategy
from .ml import resolve_pending_strategies


def simulate_well_mixed(game_name, pop_cfg, rl_params, il_params, ml_params):
    payoff_matrix = get_payoff_matrix(game_name)
    rng = make_rng(pop_cfg.seed)
    strategies = initialize_strategies(pop_cfg.N, pop_cfg.initial_cooperator_fraction, rng)
    rl_state = ReinforcementState.initialize(pop_cfg.N, rl_params.H0, rl_params.F0_C, rl_params.F0_D)

    mfc = np.zeros(ml_params.max_steps + 1, dtype=float)
    u_c_hist = np.zeros(ml_params.max_steps + 1, dtype=float)
    u_d_hist = np.zeros(ml_params.max_steps + 1, dtype=float)

    mfc[0] = fraction_cooperators(strategies)
    u_c_hist[0], u_d_hist[0] = well_mixed_expected_payoffs(strategies, payoff_matrix)

    for t in range(1, ml_params.max_steps + 1):
        pending_il = il_pending_well_mixed(strategies, payoff_matrix, il_params.eta, rng)
        payoff_if_c, payoff_if_d = well_mixed_counterfactual_payoffs(strategies, payoff_matrix)
        pending_rl = rl_pending_strategy(strategies, payoff_if_c, payoff_if_d, rl_state, rl_params, rng)
        strategies = resolve_pending_strategies(pending_rl, pending_il, ml_params.theta, rng)
        mfc[t] = fraction_cooperators(strategies)
        u_c_hist[t], u_d_hist[t] = well_mixed_expected_payoffs(strategies, payoff_matrix)

    return {
        "mfc": mfc,
        "u_c": u_c_hist,
        "u_d": u_d_hist,
        "final_strategies": strategies,
        "metadata": {
            "game": game_name,
            "population": asdict(pop_cfg),
            "rl": asdict(rl_params),
            "il": asdict(il_params),
            "ml": asdict(ml_params),
        },
    }


def simulate_square_lattice(game_name, rows, cols, pop_cfg, rl_params, il_params, ml_params):
    payoff_matrix = get_payoff_matrix(game_name)
    rng = make_rng(pop_cfg.seed)
    graph = build_square_lattice_graph(rows, cols)
    neighbors = neighbor_lists_from_graph(graph)
    N = rows * cols
    strategies = initialize_strategies(N, pop_cfg.initial_cooperator_fraction, rng)
    rl_state = ReinforcementState.initialize(N, rl_params.H0, rl_params.F0_C, rl_params.F0_D)

    mfc = np.zeros(ml_params.max_steps + 1, dtype=float)
    mfc[0] = fraction_cooperators(strategies)
    snapshots = {0: strategies.reshape(rows, cols).copy()}

    for t in range(1, ml_params.max_steps + 1):
        pending_il = il_pending_lattice(strategies, payoff_matrix, neighbors, il_params.eta, rng)
        payoff_if_c, payoff_if_d = lattice_counterfactual_payoffs(strategies, payoff_matrix, neighbors)
        pending_rl = rl_pending_strategy(strategies, payoff_if_c, payoff_if_d, rl_state, rl_params, rng)
        strategies = resolve_pending_strategies(pending_rl, pending_il, ml_params.theta, rng)
        mfc[t] = fraction_cooperators(strategies)
        if t in {50, 100, 150, 200}:
            snapshots[t] = strategies.reshape(rows, cols).copy()

    return {
        "mfc": mfc,
        "final_strategies": strategies,
        "snapshots": snapshots,
        "metadata": {
            "game": game_name,
            "rows": rows,
            "cols": cols,
            "population": asdict(pop_cfg),
            "rl": asdict(rl_params),
            "il": asdict(il_params),
            "ml": asdict(ml_params),
        },
    }
