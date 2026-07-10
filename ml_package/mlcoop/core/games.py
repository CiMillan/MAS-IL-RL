import numpy as np

STRATEGY_C = 1
STRATEGY_D = 0

PAYOFFS = {
    "PDG": np.array([[1.0, 0.0], [10.0, 8.0]], dtype=float),
    "CG": np.array([[2.0, 0.0], [4.0, 1.0]], dtype=float),
    "CoG": np.array([[3.0, 1.0], [0.0, 2.0]], dtype=float),
}


def get_payoff_matrix(name: str) -> np.ndarray:
    if name not in PAYOFFS:
        raise ValueError(f"Unknown game: {name}. Expected one of {list(PAYOFFS)}")
    return PAYOFFS[name]
