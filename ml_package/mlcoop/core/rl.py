from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import softmax_binary


@dataclass
class ReinforcementState:
    F_C: np.ndarray
    F_D: np.ndarray
    H: np.ndarray

    @classmethod
    def initialize(cls, N, H0, F0_C, F0_D):
        return cls(
            F_C=np.full(N, F0_C, dtype=float),
            F_D=np.full(N, F0_D, dtype=float),
            H=np.full(N, H0, dtype=float),
        )


def rl_pending_strategy(strategies, payoff_if_c, payoff_if_d, rl_state, params, rng):
    chosen_c = strategies == 1
    chosen_d = ~chosen_c
    realized = np.where(chosen_c, payoff_if_c, payoff_if_d)

    # AMBIGUITY FROM PAPER:
    # Eq. 5 is garbled in the extracted PDF. This block uses a standard EWA-style
    # reconstruction consistent with the article's verbal description:
    # - prior attraction is discounted,
    # - chosen action gets realized payoff,
    # - unchosen action gets delta-weighted counterfactual payoff,
    # - experience weight H is updated each round.
    # If figure reproduction is poor, revise THIS function first.
    new_H = (1.0 - params.rho) * rl_state.H + 1.0
    numer_c = params.phi * rl_state.H * rl_state.F_C + params.alpha * np.where(
        chosen_c, realized, params.delta * payoff_if_c
    )
    numer_d = params.phi * rl_state.H * rl_state.F_D + params.alpha * np.where(
        chosen_d, realized, params.delta * payoff_if_d
    )

    rl_state.F_C = numer_c / new_H
    rl_state.F_D = numer_d / new_H
    rl_state.H = new_H

    p_c = softmax_binary(rl_state.F_C, rl_state.F_D, params.beta)
    return (rng.random(len(strategies)) < p_c).astype(np.int8)
