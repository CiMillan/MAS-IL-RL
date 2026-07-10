# mlcoop

A clean, package-style Python implementation scaffold for the mixed learning (ML) cooperation paper.

This package is organized so the simulation logic you likely **should not touch often** is isolated inside `mlcoop/core/`, while experiment scripts and parameter sweeps live in `mlcoop/experiments/`.

## Design goal

The paper clearly specifies the game types, topology choices, high-level ML workflow, and baseline parameters. However, some low-level notation in the RL update rule is garbled in PDF extraction, so this package isolates all such choices behind explicit functions and **AMBIGUITY FROM PAPER** comments.

## Package layout

- `mlcoop/core/games.py` — payoff matrices and helpers
- `mlcoop/core/config.py` — dataclasses for parameters
- `mlcoop/core/topology.py` — well-mixed and lattice graph helpers
- `mlcoop/core/payoffs.py` — payoff calculations
- `mlcoop/core/il.py` — imitation-learning updates
- `mlcoop/core/rl.py` — reinforcement-learning updates
- `mlcoop/core/ml.py` — mixed-learning conflict resolution
- `mlcoop/core/simulators.py` — high-level simulation drivers
- `mlcoop/experiments/baselines.py` — ready-to-run baseline experiments

## What is explicit in the paper

These are treated as fixed unless you have a strong reason to change them:

- Strategies are binary: cooperation (1) and defection (0).
- Games: PDG, CG, CoG with the paper's reported payoff matrices.
- Well-mixed and square-lattice environments are both supported.
- ML uses both RL and IL pending strategies, then resolves disagreement probabilistically.
- Baseline experiments use `N=10000`, `H0=3`, `FC0=FD0=3`, and reported parameter combinations such as `0.5, 0.3, 0.01, 0.01, 0.8`.

## What is isolated because it may need adjustment

These parts are explicitly marked in code:

1. The exact EWA-style RL attraction update formula.
2. The exact definition of RL “pure payoff” in the well-mixed case.
3. Whether the trade-off coefficient governs only RL-vs-IL conflict resolution or also rule selection.
4. Which opposite-strategy neighbor is imitated in the lattice IL case.
5. Whether the IL active-learning fraction is implemented as Bernoulli thinning or exact-size sampling.

## Suggested workflow

1. Do not edit anything in `mlcoop/core/` until you have evidence a specific ambiguity needs revision.
2. Run `mlcoop/experiments/baselines.py` first.
3. Compare trajectories against the paper's reported qualitative outcomes.
4. If results differ, revise only the blocks labeled `AMBIGUITY FROM PAPER`.

## Example

```python
from mlcoop.experiments.baselines import run_well_mixed_baseline

result = run_well_mixed_baseline("PDG", seed=42)
print(result["mfc"][-1])
```
