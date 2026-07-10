"""
Research-clean single-file implementation scaffold for:
"Evolution of cooperation guided by the coexistence of imitation learning and reinforcement learning"

This version is intentionally written as ONE FILE so you can inspect everything in one place.
It uses:
- NumPy for vectorized state handling
- NetworkX for square-lattice graph construction

IMPORTANT
---------
The paper is clear about:
- the three games (PDG, CG, CoG),
- the well-mixed and square-lattice environments,
- the high-level IL / RL / ML workflow,
- the baseline parameter values used in experiments.

However, the extracted PDF text is garbled in several low-level places, especially
around the RL attraction update equation (Eq. 5) and some operational details.

For that reason, every uncertain implementation choice is marked with:
    AMBIGUITY FROM PAPER

Those markers identify exactly which blocks you should revisit first if the output
fails to match the figures or qualitative trends reported in the article.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx


# =============================================================================
# 1. CONSTANTS AND PAYOFF MATRICES
# =============================================================================
# Strategy encoding used throughout the file:
#   1 -> Cooperation (C)
#   0 -> Defection (D)
#
# Payoff matrix convention:
#             Opponent
#             C       D
# Self C   [ a11    a12 ]
# Self D   [ a21    a22 ]
# =============================================================================

STRATEGY_C = 1
STRATEGY_D = 0

PAYOFFS: Dict[str, np.ndarray] = {
    "PDG": np.array([[1.0, 0.0], [10.0, 8.0]], dtype=float),
    "CG":  np.array([[2.0, 0.0], [4.0, 1.0]], dtype=float),
    "CoG": np.array([[3.0, 1.0], [0.0, 2.0]], dtype=float),
}


# =============================================================================
# 2. CONFIGURATION DATACLASSES
# =============================================================================
# These keep all inputs explicit and easy to compare to the paper.
# =============================================================================

@dataclass(frozen=True)
class RLParams:
    """
    Parameters used in the reinforcement-learning component.

    Notes
    -----
    The paper clearly names these parameters and gives baseline values, but Eq. 5
    is not fully recoverable from the extracted PDF text. So these parameters are
    faithfully exposed, while the exact update algebra is isolated in one function.
    """
    alpha: float = 0.3
    phi: float = 0.01
    rho: float = 0.01
    delta: float = 0.8
    beta: float = 1.0
    H0: float = 3.0
    F0_C: float = 3.0
    F0_D: float = 3.0


@dataclass(frozen=True)
class ILParams:
    """Parameters for imitation learning."""
    eta: float = 0.5


@dataclass(frozen=True)
class MLParams:
    """Parameters for the mixed learning rule."""
    theta: float = 0.5
    max_steps: int = 100


@dataclass(frozen=True)
class PopulationConfig:
    """Population configuration."""
    N: int = 10_000
    initial_cooperator_fraction: float = 0.5
    seed: Optional[int] = 42


# =============================================================================
# 3. BASIC HELPERS
# =============================================================================


def make_rng(seed: Optional[int]) -> np.random.Generator:
    """Create a reproducible random number generator."""
    return np.random.default_rng(seed)



def initialize_strategies(N: int, coop_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Create the initial binary strategy vector."""
    return (rng.random(N) < coop_fraction).astype(np.int8)



def mean_fraction_of_cooperators(strategies: np.ndarray) -> float:
    """Compute the current mean fraction of cooperators (MFC)."""
    return float(np.mean(strategies))



def softmax_binary(value_c: np.ndarray, value_d: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute the probability of choosing cooperation from two attractions.

    p(C) = exp(beta * F_C) / (exp(beta * F_C) + exp(beta * F_D))
    """
    m = np.maximum(beta * value_c, beta * value_d)
    exp_c = np.exp(beta * value_c - m)
    exp_d = np.exp(beta * value_d - m)
    return exp_c / (exp_c + exp_d)


# =============================================================================
# 4. GAME ACCESS
# =============================================================================


def get_payoff_matrix(game_name: str) -> np.ndarray:
    """Return the payoff matrix for one of the paper's three games."""
    if game_name not in PAYOFFS:
        raise ValueError(f"Unknown game '{game_name}'. Expected one of {list(PAYOFFS)}")
    return PAYOFFS[game_name]


# =============================================================================
# 5. TOPOLOGY
# =============================================================================
# We use NetworkX for the square lattice so the topology logic stays readable.
# =============================================================================


def build_square_lattice_graph(rows: int, cols: int) -> nx.Graph:
    """
    Build the square lattice network used in the paper.

    Boundary nodes have degree 2 or 3, interior nodes have degree 4.
    """
    g = nx.grid_2d_graph(rows, cols)
    return nx.convert_node_labels_to_integers(g, ordering="sorted")



def neighbor_lists_from_graph(graph: nx.Graph) -> List[List[int]]:
    """Convert a NetworkX graph into a plain neighbor-list structure."""
    return [list(graph.neighbors(i)) for i in range(graph.number_of_nodes())]


# =============================================================================
# 6. WELL-MIXED PAYOFFS
# =============================================================================
# These formulas are stated clearly in the paper.
# =============================================================================


def well_mixed_expected_payoffs(strategies: np.ndarray, payoff_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Compute expected payoffs for C and D in a well-mixed population.

    Paper equations:
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



def well_mixed_counterfactual_payoffs(strategies: np.ndarray, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the RL payoffs for choosing C and choosing D in the well-mixed setting.

    AMBIGUITY FROM PAPER
    --------------------
    The extracted text says that under RL, "pure payoffs" are derived from the payoff
    matrix and opponents, but it does not fully specify the operational choice in the
    well-mixed case.

    This implementation uses expected payoff against the current population mix.
    That is the cleanest deterministic interpretation.

    Alternative interpretations to test later:
    - sample one opponent per agent each round,
    - sample multiple opponents and average,
    - use agent-specific interaction histories.
    """
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)
    N = len(strategies)
    return np.full(N, u_c, dtype=float), np.full(N, u_d, dtype=float)


# =============================================================================
# 7. LATTICE PAYOFFS
# =============================================================================
# The lattice average payoff definition is clear in the paper.
# =============================================================================


def lattice_realized_average_payoffs(
    strategies: np.ndarray,
    payoff_matrix: np.ndarray,
    neighbors: List[List[int]],
) -> np.ndarray:
    """Compute each agent's average payoff against its fixed neighbors."""
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



def lattice_counterfactual_payoffs(
    strategies: np.ndarray,
    payoff_matrix: np.ndarray,
    neighbors: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the payoff each lattice agent would get from choosing C or D against
    its current neighborhood.

    This is the clean local counterfactual used by the RL block.
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
# 8. IMITATION LEARNING (IL)
# =============================================================================
# The high-level IL logic is stated clearly, but some micro-choices are not.
# =============================================================================


def il_pending_well_mixed(
    strategies: np.ndarray,
    payoff_matrix: np.ndarray,
    eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute the pending IL strategy for each agent in a well-mixed population.

    Logic from the paper:
    - compare expected payoff of C versus D,
    - identify agents using the lower-payoff strategy,
    - randomly select a ratio eta of them to switch.
    """
    pending = strategies.copy()
    u_c, u_d = well_mixed_expected_payoffs(strategies, payoff_matrix)

    if np.isclose(u_c, u_d):
        return pending

    if u_c > u_d:
        eligible = np.where(strategies == 0)[0]
        target_strategy = 1
    else:
        eligible = np.where(strategies == 1)[0]
        target_strategy = 0

    # AMBIGUITY FROM PAPER
    # --------------------
    # The paper says a ratio eta of eligible agents is randomly selected, but it does
    # not force one unique sampling rule.
    #
    # This implementation uses Bernoulli thinning:
    #   each eligible agent updates independently with probability eta.
    #
    # Alternative to test later:
    #   sample exactly floor(eta * number_of_eligible_agents).
    selected = eligible[rng.random(len(eligible)) < eta]
    pending[selected] = target_strategy
    return pending



def il_pending_lattice(
    strategies: np.ndarray,
    payoff_matrix: np.ndarray,
    neighbors: List[List[int]],
    eta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute the pending IL strategy for each agent in a square lattice.

    Logic from the paper:
    - if agent and all neighbors use the same strategy, agent stays unchanged,
    - otherwise compare the agent's average payoff to neighbors with the opposite strategy,
    - if the agent has the minimal payoff among that comparison set, it may adjust,
    - then randomly select a ratio eta of eligible agents to update.
    """
    pending = strategies.copy()
    payoffs = lattice_realized_average_payoffs(strategies, payoff_matrix, neighbors)
    eligible = []
    targets = {}

    for i, nbrs in enumerate(neighbors):
        my_strategy = strategies[i]
        opposite_neighbors = [j for j in nbrs if strategies[j] != my_strategy]

        if not opposite_neighbors:
            continue

        opposite_payoffs = np.array([payoffs[j] for j in opposite_neighbors], dtype=float)

        if payoffs[i] <= np.min(opposite_payoffs):
            eligible.append(i)

            # AMBIGUITY FROM PAPER
            # --------------------
            # The article says the agent may adjust after comparing against neighbors
            # with the opposite strategy, but the extracted text does not fully specify
            # which opposite-strategy neighbor is imitated.
            #
            # This implementation chooses the highest-payoff opposite-strategy neighbor.
            # Alternatives to test later:
            # - imitate a random better opposite-strategy neighbor,
            # - imitate the best overall neighbor regardless of tie handling,
            # - switch directly to the opposite strategy without selecting a neighbor.
            best_neighbor = opposite_neighbors[int(np.argmax(opposite_payoffs))]
            targets[i] = strategies[best_neighbor]

    eligible = np.array(eligible, dtype=int)

    # AMBIGUITY FROM PAPER
    # --------------------
    # Same sampling ambiguity as the well-mixed IL case.
    selected = eligible[rng.random(len(eligible)) < eta] if len(eligible) else eligible

    for i in selected:
        pending[i] = targets[i]

    return pending


# =============================================================================
# 9. REINFORCEMENT LEARNING (RL)
# =============================================================================
# This is the main uncertainty zone because Eq. 5 is garbled in extraction.
# We therefore keep the full RL update in one isolated block.
# =============================================================================

@dataclass
class ReinforcementState:
    F_C: np.ndarray
    F_D: np.ndarray
    H: np.ndarray

    @classmethod
    def initialize(cls, N: int, params: RLParams) -> "ReinforcementState":
        return cls(
            F_C=np.full(N, params.F0_C, dtype=float),
            F_D=np.full(N, params.F0_D, dtype=float),
            H=np.full(N, params.H0, dtype=float),
        )



def rl_pending_strategy(
    strategies: np.ndarray,
    payoff_if_c: np.ndarray,
    payoff_if_d: np.ndarray,
    rl_state: ReinforcementState,
    params: RLParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Update attractions and produce the RL pending strategy.

    AMBIGUITY FROM PAPER
    --------------------
    Eq. 5 in the extracted PDF is not fully readable. The verbal description says:
    - preferences/attractions are updated for both strategies,
    - the selected strategy uses realized payoff,
    - the unselected strategy uses an assumed payoff weighted by delta,
    - H tracks experience / observation-equivalents,
    - strategy choice is probabilistic via Eq. 6.

    This implementation uses a standard EWA-style reconstruction consistent with
    that description.

    If your output differs substantially from the paper, revise THIS function first.
    """
    chosen_c = strategies == 1
    chosen_d = ~chosen_c

    realized = np.where(chosen_c, payoff_if_c, payoff_if_d)

    # AMBIGUITY FROM PAPER
    # --------------------
    # Exact H update and attraction algebra are reconstructed here in a clean,
    # standard EWA-like form.
    new_H = (1.0 - params.rho) * rl_state.H + 1.0

    numer_c = (
        params.phi * rl_state.H * rl_state.F_C
        + params.alpha * np.where(chosen_c, realized, params.delta * payoff_if_c)
    )
    numer_d = (
        params.phi * rl_state.H * rl_state.F_D
        + params.alpha * np.where(chosen_d, realized, params.delta * payoff_if_d)
    )

    rl_state.F_C = numer_c / new_H
    rl_state.F_D = numer_d / new_H
    rl_state.H = new_H

    p_c = softmax_binary(rl_state.F_C, rl_state.F_D, params.beta)
    return (rng.random(len(strategies)) < p_c).astype(np.int8)


# =============================================================================
# 10. MIXED LEARNING (ML)
# =============================================================================
# The conflict-resolution idea is clear, but one detail remains ambiguous:
# whether theta is only the final conflict weight or also an ex ante rule-selector.
# =============================================================================


def resolve_pending_strategies(
    pending_rl: np.ndarray,
    pending_il: np.ndarray,
    theta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resolve RL and IL pending strategies into the next-period strategy vector.

    Logic used here:
    - if RL and IL agree, keep that strategy,
    - if they disagree, choose RL with probability theta, IL with probability 1-theta.
    """
    # AMBIGUITY FROM PAPER
    # --------------------
    # The extracted text strongly supports using theta for RL-vs-IL conflict resolution.
    # It is less clear whether the same parameter should ALSO govern a prior rule-choice
    # step before conflict resolution.
    #
    # This implementation uses theta ONLY here, at the disagreement-resolution stage.
    same = pending_rl == pending_il
    final = pending_rl.copy()
    conflict = np.where(~same)[0]

    if len(conflict) == 0:
        return final.astype(np.int8)

    choose_rl = rng.random(len(conflict)) < theta
    final[conflict[~choose_rl]] = pending_il[conflict[~choose_rl]]
    return final.astype(np.int8)


# =============================================================================
# 11. WELL-MIXED SIMULATION DRIVER
# =============================================================================


def simulate_well_mixed(
    game_name: str,
    pop_cfg: PopulationConfig,
    rl_params: RLParams,
    il_params: ILParams,
    ml_params: MLParams,
) -> Dict[str, np.ndarray]:
    """Run the mixed-learning model in a well-mixed population."""
    payoff_matrix = get_payoff_matrix(game_name)
    rng = make_rng(pop_cfg.seed)
    strategies = initialize_strategies(pop_cfg.N, pop_cfg.initial_cooperator_fraction, rng)
    rl_state = ReinforcementState.initialize(pop_cfg.N, rl_params)

    mfc = np.zeros(ml_params.max_steps + 1, dtype=float)
    u_c_hist = np.zeros(ml_params.max_steps + 1, dtype=float)
    u_d_hist = np.zeros(ml_params.max_steps + 1, dtype=float)

    mfc[0] = mean_fraction_of_cooperators(strategies)
    u_c_hist[0], u_d_hist[0] = well_mixed_expected_payoffs(strategies, payoff_matrix)

    for t in range(1, ml_params.max_steps + 1):
        pending_il = il_pending_well_mixed(strategies, payoff_matrix, il_params.eta, rng)
        payoff_if_c, payoff_if_d = well_mixed_counterfactual_payoffs(strategies, payoff_matrix)
        pending_rl = rl_pending_strategy(strategies, payoff_if_c, payoff_if_d, rl_state, rl_params, rng)
        strategies = resolve_pending_strategies(pending_rl, pending_il, ml_params.theta, rng)

        mfc[t] = mean_fraction_of_cooperators(strategies)
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


# =============================================================================
# 12. LATTICE SIMULATION DRIVER
# =============================================================================


def simulate_square_lattice(
    game_name: str,
    rows: int,
    cols: int,
    pop_cfg: PopulationConfig,
    rl_params: RLParams,
    il_params: ILParams,
    ml_params: MLParams,
) -> Dict[str, np.ndarray]:
    """Run the mixed-learning model on a square lattice network."""
    payoff_matrix = get_payoff_matrix(game_name)
    rng = make_rng(pop_cfg.seed)

    graph = build_square_lattice_graph(rows, cols)
    neighbors = neighbor_lists_from_graph(graph)

    N = rows * cols
    strategies = initialize_strategies(N, pop_cfg.initial_cooperator_fraction, rng)
    rl_state = ReinforcementState.initialize(N, rl_params)

    mfc = np.zeros(ml_params.max_steps + 1, dtype=float)
    mfc[0] = mean_fraction_of_cooperators(strategies)
    snapshots = {0: strategies.reshape(rows, cols).copy()}

    for t in range(1, ml_params.max_steps + 1):
        pending_il = il_pending_lattice(strategies, payoff_matrix, neighbors, il_params.eta, rng)
        payoff_if_c, payoff_if_d = lattice_counterfactual_payoffs(strategies, payoff_matrix, neighbors)
        pending_rl = rl_pending_strategy(strategies, payoff_if_c, payoff_if_d, rl_state, rl_params, rng)
        strategies = resolve_pending_strategies(pending_rl, pending_il, ml_params.theta, rng)

        mfc[t] = mean_fraction_of_cooperators(strategies)
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


# =============================================================================
# 13. PAPER-STYLE BASELINES
# =============================================================================
# These use the parameter values reported in the paper's ML experiments.
# =============================================================================


def paper_baseline_well_mixed(game_name: str, seed: Optional[int] = 42):
    pop_cfg = PopulationConfig(N=10_000, initial_cooperator_fraction=0.5, seed=seed)
    rl_params = RLParams(alpha=0.3, phi=0.01, rho=0.01, delta=0.8, beta=1.0, H0=3.0, F0_C=3.0, F0_D=3.0)
    il_params = ILParams(eta=0.5)
    ml_params = MLParams(theta=0.5, max_steps=100)
    return simulate_well_mixed(game_name, pop_cfg, rl_params, il_params, ml_params)



def paper_baseline_lattice(game_name: str, seed: Optional[int] = 42):
    pop_cfg = PopulationConfig(N=10_000, initial_cooperator_fraction=0.5, seed=seed)
    rl_params = RLParams(alpha=0.3, phi=0.01, rho=0.01, delta=0.8, beta=1.0, H0=3.0, F0_C=3.0, F0_D=3.0)
    il_params = ILParams(eta=0.5)
    ml_params = MLParams(theta=0.5, max_steps=200)
    return simulate_square_lattice(game_name, rows=100, cols=100, pop_cfg=pop_cfg, rl_params=rl_params, il_params=il_params, ml_params=ml_params)


# =============================================================================
# 14. MINIMAL RUNNER
# =============================================================================
# This makes the file directly runnable.
# =============================================================================

if __name__ == "__main__":
    print("Running paper-style well-mixed baselines...")
    for game in ["PDG", "CG", "CoG"]:
        result = paper_baseline_well_mixed(game, seed=42)
        print(f"{game} final MFC: {result['mfc'][-1]:.4f}")

    print("\nNotes:")
    print("- If the results do not resemble the paper, inspect every 'AMBIGUITY FROM PAPER' block.")
    print("- The RL update function is the highest-priority revision point.")
