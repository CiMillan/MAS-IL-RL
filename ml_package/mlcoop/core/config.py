from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RLParams:
    phi: float = 0.8
    rho: float = 0.01
    delta: float = 0.01
    beta: float = 1.0
    H0: float = 3.0
    F0_C: float = 3.0
    F0_D: float = 3.0


@dataclass(frozen=True)
class ILParams:
    eta: float = 0.5


@dataclass(frozen=True)
class MLParams:
    theta: float = 0.5
    max_steps: int = 100


@dataclass(frozen=True)
class PopulationConfig:
    N: int = 10000
    initial_cooperator_fraction: float = 0.5
    seed: Optional[int] = 42
