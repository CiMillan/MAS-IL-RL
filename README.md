# Coexistence of Imitation Learning and Reinforcement Learning in Cooperation Evolution

This repository contains a Python-based multi-agent simulation framework replicating and exploring the dynamics described in the paper:

> **Evolution of cooperation guided by the coexistence of imitation learning and reinforcement learning**  
> *Wei Tang, Guoling Wang, and Zhiyan Xing*  
> Published in *Scientific Reports* (2025), Nature Portfolio  
> [Search on Google Scholar](https://scholar.google.com/scholar?q=Evolution+of+cooperation+guided+by+the+coexistence+of+imitation+learning+and+reinforcement+learning&hl=en&as_sdt=0,5)

---

## 🔬 Scientific Overview

In evolutionary game theory, cooperation is traditionally modeled using **Imitation Learning (IL)**, where agents adopt strategies from successful neighbors. This research introduces a hybrid model combining IL with **Reinforcement Learning (RL)**, where agents also adapt autonomously based on personal experience and feedback.

The coexistence of both paradigms is studied across three classic games:
1. **Prisoner's Dilemma Game (PDG)**
2. **Coordination Game (CG)**
3. **Coexistence Game (CoG) / Hawk-Dove / Snowdrift**

Simulations are executed across two population structures:
* **Well-mixed populations** (fully connected networks)
* **Square lattices** (grid topologies where interactions are local)

---

## 📁 Repository Structure

The project is structured to offer both a simple, self-contained single-file script and a modular, clean package:

* **[main.py](file:///Users/cintiamillan/Desktop/MAS-IL-RL/main.py)**: A research-clean, single-file implementation scaffold containing all models, payoff configurations, topologies, and simulation drivers in one place.
* **[ml_package/](file:///Users/cintiamillan/Desktop/MAS-IL-RL/ml_package)**: A modularized version of the simulation framework package (`mlcoop`):
  * `mlcoop/core/config.py`: Dataclasses for RL, IL, and ML parameters.
  * `mlcoop/core/games.py`: Game payoff matrices (PDG, CG, CoG).
  * `mlcoop/core/topology.py`: Graph creation (square lattices) and neighborhood mappings.
  * `mlcoop/core/payoffs.py`: Average and counterfactual payoff calculations.
  * `mlcoop/core/il.py` / `rl.py` / `ml.py`: Step update rules for imitation, reinforcement, and conflict resolution.
  * `mlcoop/core/simulators.py`: High-level simulation runners.
  * `mlcoop/experiments/baselines.py`: Scripts to run baseline sweeps.
* **[mainV1/](file:///Users/cintiamillan/Desktop/MAS-IL-RL/mainV1)**: Legacy/v1 version of the single-file simulator.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:CiMillan/MAS-IL-RL.git
   cd MAS-IL-RL
   ```

2. **Install dependencies**:
   This simulation relies on standard scientific Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Requires `numpy`, `networkx`, and optionally `matplotlib` for plotting).*

---

## 🚀 Running Simulations

### 1. Single-File Scaffold
To run the self-contained baseline simulations:
```bash
python3 main.py
```

### 2. Modular Package
To run or extend using the modular package structure:
```python
from ml_package.mlcoop.experiments.baselines import run_well_mixed_baseline

# Run a baseline simulation for the Prisoner's Dilemma
results = run_well_mixed_baseline("PDG", seed=42)
print("Final MFC (Mean Fraction of Cooperators):", results["mfc"][-1])
```

---

## ⚠️ Handling Paper Ambiguities

To make the codebase robust and easy to tune, we have isolated these assumptions behind explicit comments marked as:
`# AMBIGUITY FROM PAPER`

Key areas with assumptions:
1. **EWA attraction updates**: The specific update algebra in `rl_pending_strategy` uses a standard, robust Experience-Weighted Attraction (EWA) model.
2. **Well-mixed counterfactuals**: Defining "pure payoffs" in a well-mixed setting is implemented using expected payoffs against the global population ratio.
3. **Imitation selection**: In the lattice model, if multiple neighbors use the opposite strategy, the agent imitates the neighbor with the highest average payoff.
4. **Bernoulli thinning**: The learning update ratio ($\eta$) is modeled as independent probability updates per step (Bernoulli trials).
