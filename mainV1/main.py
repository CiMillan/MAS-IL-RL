"""
Replication code for:
Evolution of cooperation guided by the coexistence of imitation learning and reinforcement learning

This file is written as a heavily documented implementation intended for study and
step-by-step reasoning, not just fast execution. It reproduces the model structure
reported in the paper and provides a clean simulation scaffold for both:

1. Well-mixed populations
2. Square lattice populations

Important note
--------------

This code is best understood as a transparent replication scaffold:
- the payoff matrices match the paper exactly,
- the RL and IL update logic follows the paper's description,
- the ML combination rule follows the paper's step-by-step procedure,
- experiments are parameterized so you can reproduce and then tune details if needed.

Recommended use
---------------
Start with the well-mixed model first, verify the qualitative trends, and only then
move to the square lattice model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# 1. PAYOFF MATRICES
# =============================================================================
# The paper uses a symmetric 2x2 game with two actions:
# - C = cooperation
# - D = defection
#
# We encode strategies as integers for efficient computation:
# - 1 means C
# - 0 means D
#
# Matrix convention:
#          Opponent
#          C     D
# Self C [a11,  a12]
# Self D [a21,  a22]
#
# So if I cooperate and my opponent defects, I receive a12.
# If I defect and my opponent cooperates, I receive a21.
# =============================================================================

PAYOFFS: Dict[str, np.ndarray] = {
    "PDG": np.array([[1.0, 0.0],
                     [10.0, 8.0]]),
    "CG":  np.array([[2.0, 0.0],
                     [4.0, 1.0]]),
    "CoG": np.array([[3.0, 1.0],
                     [0.0, 2.0]]),
}


# =============================================================================
# 2. CONFIGURATION OBJECTS
# =============================================================================
# Using dataclasses makes the simulation parameters explicit and easy to inspect.
# =============================================================================

@dataclass
class RLParams:
    """
    Parameters for reinforcement learning (EWA-style) updates.

    Attributes
    ----------
    alpha : float
        Introspection rate in the paper's language. In this implementation it controls
        how strongly realized/assumed payoffs enter the attractions. A higher value
        means attractions react more strongly to recent outcomes.

    phi : float
        Discount factor on previous attractions. Larger values preserve past preference
        information more strongly.

    rho : float
        Depreciation rate for the experience weight H_t. This controls how much past
        experience remains relevant relative to the new round.

    delta : float
        Weight placed on the unselected strategy's assumed payoff, following the EWA
        description in the paper.

    beta : float
        Softmax sensitivity. Higher beta means the agent chooses the higher-attraction
        strategy more deterministically.

    H0 : float
        Initial experience weight.

    F0_C : float
        Initial attraction for cooperation.

    F0_D : float
        Initial attraction for defection.
    """
    alpha: float = 0.3
    phi: float = 0.01
    rho: float = 0.01
    delta: float = 0.8
    beta: float = 1.0
    H0: float = 3.0
    F0_C: float = 3.0
    F0_D: float = 3.0


@dataclass
class ILParams:
    """
    Parameters for imitation learning.

    Attributes
    ----------
    eta : float
        Fraction of eligible agents who actually revise under the imitation rule.
        The paper describes selecting a ratio of agents among those with lower payoff.
        This parameter is that ratio.
    """
    eta: float = 0.5


@dataclass
class MLParams:
    """
    Parameters for the mixed learning rule.

    Attributes
    ----------
    mix_rule_prob : float
        Probability of selecting the RL rule when forming pending strategy 1.
        The IL rule is then selected with probability (1 - mix_rule_prob).
        In the paper this is the probability of choosing RL.

    conflict_bias_rl : float
        When RL and IL propose different strategies, choose the RL proposal with this
        probability and the IL proposal with probability (1 - conflict_bias_rl).

    max_steps : int
        Number of generations/iterations.
    """
    mix_rule_prob: float = 0.5
    conflict_bias_rl: float = 0.5
    max_steps: int = 100


@dataclass
class PopulationConfig:
    """
    General population configuration.
    """
    N: int = 10_000
    initial_cooperator_fraction: float = 0.5
    seed: Optional[int] = 42


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def make_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a NumPy random generator so results are reproducible."""
    return np.random.default_rng(seed)


def initialize_strategies(N: int, coop_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """
    Create the initial population strategy vector.

    Returns
    -------
    strategies : ndarray of shape (N,)
        Binary vector: 1 for cooperation, 0 for defection.
    """
    return (rng.random(N) < coop_fraction).astype(np.int8)


def softmax_binary(value_c: np.ndarray, value_d: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute P(C) from two attraction values using a stable binary softmax.

    P(C) = exp(beta * F_C) / (exp(beta * F_C) + exp(beta * F_D))
    """
    stacked = np.vstack([beta * value_c, beta * value_d])
    max_vals = np.max(stacked, axis=0)
    exp_c = np.exp(beta * value_c - max_vals)
    exp_d = np.exp(beta * value_d - max_vals)
    return exp_c / (exp_c + exp_d)


def mean_fraction_of_cooperators(strategies: np.ndarray) -> float:
    """Return the current proportion of cooperators in the population."""
    return float(np.mean(strategies))


# =============================================================================
# 4. WELL-MIXED POPULATION PAYOFFS
# =============================================================================

def well_mixed_expected_payoffs(strategies: np.ndarray, payoff_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Compute expected payoff of choosing C and D in a well-mixed population.

    Paper-consistent formulas:
    u_C = [a11 * (x_C - 1) + a12 * x_D] / (N - 1)
    u_D = [a21 * x_C + a22 * (x_D - 1)] / (N - 1)
    """
    N = len(strategies)
    x_c = int(np.sum(strategies))
    x_d = N - x_c

    a11, a12 = payoff_matrix[0, 0], payoff_matrix[0, 1]
    a21, a22 = payoff_matrix[1, 0], payoff_matrix[1, 1]

    if N <= 1:
        return 0.0, 0.0

    u_c = (a11 * max(x_c - 1, 0) + a12 * x_d) / (N - 1)
    u_d = (a21 * x_c + a22 * max(x_d - 1, 0)) / (N - 1)
    return float(u_c), float(u_d)


def realized_payoffs_well_mixed(strategies: np.ndarray, payoff_matrix: np.ndarray) -> np.ndarray:
    """
    Compute one realized payoff per agent under the well-mixed expected-payoff view.

    In the well-mixed case, a transparent approximation is to assign each agent the
    expected payoff associated with its current strategy.
    """
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)
    return np.where(strategies == 1, u_c, u_d).astype(float)


def counterfactual_payoffs_well_mixed(strategies: np.ndarray, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each agent, return the payoff it would obtain from choosing C and from choosing D.
    """
    N = len(strategies)
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)
    return np.full(N, u_c, dtype=float), np.full(N, u_d, dtype=float)


# =============================================================================
# 5. SQUARE LATTICE PAYOFFS
# =============================================================================

def lattice_neighbors(index: int, rows: int, cols: int) -> List[int]:
    """
    Return the von Neumann neighbors (up, down, left, right) of a cell in a grid.
    """
    r, c = divmod(index, cols)
    nbrs = []
    if r > 0:
        nbrs.append((r - 1) * cols + c)
    if r < rows - 1:
        nbrs.append((r + 1) * cols + c)
    if c > 0:
        nbrs.append(r * cols + (c - 1))
    if c < cols - 1:
        nbrs.append(r * cols + (c + 1))
    return nbrs


def build_lattice_neighbor_list(rows: int, cols: int) -> List[List[int]]:
    """Precompute the neighbors of every lattice site for efficiency and clarity."""
    return [lattice_neighbors(i, rows, cols) for i in range(rows * cols)]


def realized_payoffs_lattice(strategies: np.ndarray, payoff_matrix: np.ndarray, neighbors: List[List[int]]) -> np.ndarray:
    """
    Compute each agent's average payoff from playing against its fixed neighbors.
    """
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


def counterfactual_payoffs_lattice(strategies: np.ndarray, payoff_matrix: np.ndarray, neighbors: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute, for every agent, the average payoff it would get from choosing C and D
    against its current neighborhood composition.
    """
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


# =============================================================================
# 6. IMITATION LEARNING (IL)
# =============================================================================

def il_pending_strategy_well_mixed(
    strategies: np.ndarray,
    payoff_matrix: np.ndarray,
    il_params: ILParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Produce the IL pending strategy for each agent in a well-mixed population.
    """
    pending = strategies.copy()
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)

    if np.isclose(u_c, u_d):
        return pending

    if u_c > u_d:
        worse_agents = np.where(strategies == 0)[0]
        better_strategy = 1
    else:
        worse_agents = np.where(strategies == 1)[0]
        better_strategy = 0

    if len(worse_agents) == 0:
        return pending

    revise_mask = rng.random(len(worse_agents)) < il_params.eta
    selected = worse_agents[revise_mask]
    pending[selected] = better_strategy
    return pending


def il_pending_strategy_lattice(
    strategies: np.ndarray,
    payoff_matrix: np.ndarray,
    neighbors: List[List[int]],
    il_params: ILParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Produce the IL pending strategy for each agent in a square lattice.
    """
    pending = strategies.copy()
    payoffs = realized_payoffs_lattice(strategies, payoff_matrix, neighbors)

    eligible = []
    target_strategy = {}

    for i, nbrs in enumerate(neighbors):
        my_s = strategies[i]
        opposite = [j for j in nbrs if strategies[j] != my_s]

        if not opposite:
            continue

        opposite_payoffs = np.array([payoffs[j] for j in opposite], dtype=float)
        if payoffs[i] <= np.min(opposite_payoffs):
            best_opposite = opposite[int(np.argmax(opposite_payoffs))]
            eligible.append(i)
            target_strategy[i] = strategies[best_opposite]

    if not eligible:
        return pending

    eligible = np.array(eligible, dtype=int)
    revise = rng.random(len(eligible)) < il_params.eta
    selected = eligible[revise]

    for i in selected:
        pending[i] = target_strategy[i]

    return pending


# =============================================================================
# 7. REINFORCEMENT LEARNING (RL / EWA STYLE)
# =============================================================================

class ReinforcementState:
    """
    Mutable RL state for all agents.
    """

    def __init__(self, N: int, params: RLParams):
        self.F_C = np.full(N, params.F0_C, dtype=float)
        self.F_D = np.full(N, params.F0_D, dtype=float)
        self.H = np.full(N, params.H0, dtype=float)


def rl_update_and_pending_strategy(
    strategies: np.ndarray,
    payoff_if_c: np.ndarray,
    payoff_if_d: np.ndarray,
    rl_state: ReinforcementState,
    rl_params: RLParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Perform one RL attraction update and return the RL pending strategy.
    """
    chosen_c = strategies == 1
    chosen_d = ~chosen_c

    realized = np.where(chosen_c, payoff_if_c, payoff_if_d)
    imagined_c = payoff_if_c
    imagined_d = payoff_if_d

    new_H = (1.0 - rl_params.rho) * rl_state.H + 1.0

    numer_c = (
        rl_params.phi * rl_state.H * rl_state.F_C
        + rl_params.alpha * np.where(chosen_c, realized, rl_params.delta * imagined_c)
    )
    numer_d = (
        rl_params.phi * rl_state.H * rl_state.F_D
        + rl_params.alpha * np.where(chosen_d, realized, rl_params.delta * imagined_d)
    )

    rl_state.F_C = numer_c / new_H
    rl_state.F_D = numer_d / new_H
    rl_state.H = new_H

    p_c = softmax_binary(rl_state.F_C, rl_state.F_D, rl_params.beta)
    pending = (rng.random(len(strategies)) < p_c).astype(np.int8)
    return pending


# =============================================================================
# 8. MIXED LEARNING (ML)
# =============================================================================

def combine_pending_strategies(
    pending_rl: np.ndarray,
    pending_il: np.ndarray,
    ml_params: MLParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Combine RL and IL proposals into the final next-round strategy vector.
    """
    same = pending_rl == pending_il
    final = pending_rl.copy()

    conflict = np.where(~same)[0]
    if len(conflict) == 0:
        return final

    choose_rl = rng.random(len(conflict)) < ml_params.conflict_bias_rl
    final[conflict[~choose_rl]] = pending_il[conflict[~choose_rl]]
    return final.astype(np.int8)


# =============================================================================
# 9. WELL-MIXED SIMULATION
# =============================================================================

def simulate_well_mixed(
    game_name: str,
    pop_cfg: PopulationConfig,
    rl_params: RLParams,
    il_params: ILParams,
    ml_params: MLParams,
) -> Dict[str, np.ndarray]:
    """
    Run the mixed learning dynamics in a well-mixed population.
    """
    payoff_matrix = PAYOFFS[game_name]
    rng = make_rng(pop_cfg.seed)

    strategies = initialize_strategies(pop_cfg.N, pop_cfg.initial_cooperator_fraction, rng)
    rl_state = ReinforcementState(pop_cfg.N, rl_params)

    history_mfc = np.zeros(ml_params.max_steps + 1, dtype=float)
    history_uc = np.zeros(ml_params.max_steps + 1, dtype=float)
    history_ud = np.zeros(ml_params.max_steps + 1, dtype=float)

    history_mfc[0] = mean_fraction_of_cooperators(strategies)
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)
    history_uc[0] = u_c
    history_ud[0] = u_d

    for t in range(1, ml_params.max_steps + 1):
        pending_il = il_pending_strategy_well_mixed(strategies, payoff_matrix, il_params, rng)

        payoff_if_c, payoff_if_d = counterfactual_payoffs_well_mixed(strategies, payoff_matrix)
        pending_rl = rl_update_and_pending_strategy(
            strategies=strategies,
            payoff_if_c=payoff_if_c,
            payoff_if_d=payoff_if_d,
            rl_state=rl_state,
            rl_params=rl_params,
            rng=rng,
        )

        strategies = combine_pending_strategies(pending_rl, pending_il, ml_params, rng)

        history_mfc[t] = mean_fraction_of_cooperators(strategies)
        u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)
        history_uc[t] = u_c
        history_ud[t] = u_d

    return {
        "mfc": history_mfc,
        "u_c": history_uc,
        "u_d": history_ud,
        "final_strategies": strategies,
    }


# =============================================================================
# 10. LATTICE SIMULATION
# =============================================================================

def simulate_lattice(
    game_name: str,
    rows: int,
    cols: int,
    rl_params: RLParams,
    il_params: ILParams,
    ml_params: MLParams,
    seed: Optional[int] = 42,
    initial_cooperator_fraction: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Run the mixed learning dynamics on a square lattice.
    """
    payoff_matrix = PAYOFFS[game_name]
    rng = make_rng(seed)

    N = rows * cols
    strategies = initialize_strategies(N, initial_cooperator_fraction, rng)
    neighbors = build_lattice_neighbor_list(rows, cols)
    rl_state = ReinforcementState(N, rl_params)

    history_mfc = np.zeros(ml_params.max_steps + 1, dtype=float)
    history_mfc[0] = mean_fraction_of_cooperators(strategies)

    snapshots = {0: strategies.reshape(rows, cols).copy()}

    for t in range(1, ml_params.max_steps + 1):
        pending_il = il_pending_strategy_lattice(strategies, payoff_matrix, neighbors, il_params, rng)

        payoff_if_c, payoff_if_d = counterfactual_payoffs_lattice(strategies, payoff_matrix, neighbors)
        pending_rl = rl_update_and_pending_strategy(
            strategies=strategies,
            payoff_if_c=payoff_if_c,
            payoff_if_d=payoff_if_d,
            rl_state=rl_state,
            rl_params=rl_params,
            rng=rng,
        )

        strategies = combine_pending_strategies(pending_rl, pending_il, ml_params, rng)
        history_mfc[t] = mean_fraction_of_cooperators(strategies)

        if t in {50, 100, 150, 200}:
            snapshots[t] = strategies.reshape(rows, cols).copy()

    return {
        "mfc": history_mfc,
        "final_strategies": strategies,
        "snapshots": snapshots,
    }


# =============================================================================
# 11. PAPER-STYLE BASELINES
# =============================================================================

def paper_baseline_well_mixed(game_name: str, seed: Optional[int] = 42) -> Dict[str, np.ndarray]:
    """
    Baseline well-mixed setup reported in the paper.
    """
    pop_cfg = PopulationConfig(N=10_000, initial_cooperator_fraction=0.5, seed=seed)
    rl_params = RLParams(alpha=0.3, phi=0.01, rho=0.01, delta=0.8, beta=1.0, H0=3.0, F0_C=3.0, F0_D=3.0)
    il_params = ILParams(eta=0.5)
    ml_params = MLParams(mix_rule_prob=0.5, conflict_bias_rl=0.5, max_steps=100)
    return simulate_well_mixed(game_name, pop_cfg, rl_params, il_params, ml_params)


def paper_baseline_lattice(game_name: str, seed: Optional[int] = 42) -> Dict[str, np.ndarray]:
    """
    Baseline square-lattice setup reported in the paper.
    """
    rl_params = RLParams(alpha=0.3, phi=0.01, rho=0.01, delta=0.8, beta=1.0, H0=3.0, F0_C=3.0, F0_D=3.0)
    il_params = ILParams(eta=0.5)
    ml_params = MLParams(mix_rule_prob=0.5, conflict_bias_rl=0.5, max_steps=200)
    return simulate_lattice(game_name, 100, 100, rl_params, il_params, ml_params, seed=seed, initial_cooperator_fraction=0.5)


# =============================================================================
# 12. SENSITIVITY ANALYSIS HELPERS
# =============================================================================

def sweep_tradeoff_well_mixed(game_name: str, tradeoff_values: List[float], seed: int = 42) -> Dict[float, np.ndarray]:
    """
    Sweep the RL conflict-bias/trade-off parameter in the well-mixed population.
    """
    results = {}
    for w in tradeoff_values:
        pop_cfg = PopulationConfig(N=10_000, initial_cooperator_fraction=0.5, seed=seed)
        rl_params = RLParams(alpha=0.3, phi=0.01, rho=0.01, delta=0.8, beta=1.0)
        il_params = ILParams(eta=0.5)
        ml_params = MLParams(mix_rule_prob=0.5, conflict_bias_rl=w, max_steps=100)
        out = simulate_well_mixed(game_name, pop_cfg, rl_params, il_params, ml_params)
        results[w] = out["mfc"]
    return results


def sweep_introspection_well_mixed(game_name: str, alpha_values: List[float], seed: int = 42) -> Dict[float, np.ndarray]:
    """
    Sweep the RL introspection parameter alpha in the well-mixed population.
    """
    results = {}
    for alpha in alpha_values:
        pop_cfg = PopulationConfig(N=10_000, initial_cooperator_fraction=0.5, seed=seed)
        rl_params = RLParams(alpha=alpha, phi=0.01, rho=0.01, delta=0.8, beta=1.0)
        il_params = ILParams(eta=0.5)
        ml_params = MLParams(mix_rule_prob=0.5, conflict_bias_rl=0.5, max_steps=100)
        out = simulate_well_mixed(game_name, pop_cfg, rl_params, il_params, ml_params)
        results[alpha] = out["mfc"]
    return results


# =============================================================================
# 13. EXAMPLE RUN
# =============================================================================

if __name__ == "__main__":
    result_pdg = paper_baseline_well_mixed("PDG", seed=42)
    print("Well-mixed PDG final MFC:", result_pdg["mfc"][-1])

    result_cg = paper_baseline_well_mixed("CG", seed=42)
    print("Well-mixed CG final MFC:", result_cg["mfc"][-1])

    result_cog = paper_baseline_well_mixed("CoG", seed=42)
    print("Well-mixed CoG final MFC:", result_cog["mfc"][-1])

    # Uncomment to test the square lattice version.
    # lattice_result = paper_baseline_lattice("PDG", seed=42)
    # print("Lattice PDG final MFC:", lattice_result["mfc"][-1])