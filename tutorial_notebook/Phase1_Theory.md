# Phase 1: Theoretical Mapping - Traffic Physics for Binary Optimization

## The Core Metaphor: Traffic Congestion as Optimization Stagnation
We map the "Low Autocorrelation Binary Sequence" (LABS) problem to a Non-Hermitian Traffic Flow model (Hatano-Nelson model).

| Traffic Concept (Q-Mobility) | LABS Challenge Equivalent |
|------------------------------|---------------------------|
| **Highway Segment** | **The Energy Landscape of the Sequence** (Hamming Graph). The "highway" represents the search space where the solver moves. |
| **Traffic Congestion** | **Local Minima**. The solver gets stuck in a state with high autocorrelation (high energy). Just as cars pile up in a jam, the probability amplitude of the solver localizes in a sub-optimal basin. |
| **Spectral Criticality** | **Stagnation Detection**. We use the **Inverse Participation Ratio (IPR)** of the solver's state to detect localization. A spike in IPR indicates the solver is "frozen" in a local minimum (a traffic jam). |
| **Hatano-Nelson Drift ($\delta$)** | **Non-Hermitian Kick**. A directional, non-reciprocal force applied to the state. It breaks detailed balance, effectively creating a "one-way street" in Hilbert space that forces the solver to flow *through* and *out* of the local minimum, preventing backtracking. |

## Why This Works
Standard quantum annealing (Hermitian) relies on symmetric tunneling. If a trap is deep, the tunneling rate is low, and the solver stays stuck (Anderson Localization).
By introducing **Non-Hermitian terms** (Drift $\delta$), we break the symmetry of the transition probabilities ($H_{i \to j} \neq H_{j \to i}$).
This induces a **Delocalization Transition**, converting the "localized" (stuck) phase into a "conducting" (flowing) phase, allowing the solver to explore new regions of the Hamming graph efficiently.

## The Mathematical "Hammer"
The effective Hamiltonian applied during the "Shockwave" phase is:
$$H_{eff} = \sum_i (t+\delta) c^\dagger_i c_{i+1} + (t-\delta) c^\dagger_{i+1} c_i$$
Where $\delta$ controls the strength of the non-reciprocal drift.
