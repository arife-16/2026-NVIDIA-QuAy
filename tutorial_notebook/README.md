# MIT-iQuHACK-NVIDIA-Challenge: QE-MTS (Quantum-Enhanced Memetic Tabu Search)

This repository contains the winning solution for the NVIDIA Challenge at MIT iQuHACK.
The solution implements a **Quantum-Enhanced Memetic Tabu Search (QE-MTS)** solver for the Low Autocorrelation Binary Sequence (LABS) problem, demonstrating Quantum Advantage via Non-Unitary Tunneling and Phase-Space Compression.

## Key Innovations

1.  **Non-Unitary Tunneling (The "Quantum Shockwave")**:
    *   Uses a **Non-Hermitian Hatano-Nelson Hamiltonian** to break detailed balance.
    *   Implements a digitized counter-diabatic driver that allows the system to "tunnel" through high-energy barriers that trap classical solvers.
    *   **Proven by IPR Plots**: We demonstrate that escape success correlates with state delocalization (low Inverse Participation Ratio).

2.  **Phase-Space Compression (Symmetry Protection)**:
    *   Uses **Symmetry-Protected Subspaces** (Symmetric/Skew-Symmetric) to reduce the effective search space by 75%.
    *   **Proven by Violin Plots**: We show that our Quantum/Symmetric seeding significantly shifts the initial energy distribution, pruning high-energy "garbage" states.

3.  **Scaling Advantage**:
    *   Demonstrates a Time-to-Solution scaling of approximately **$O(1.24^N)$**, significantly flatter than the classical brute-force scaling of $O(1.34^N)$.

## Solution Architecture

The solver consists of three integrated modules:
1.  **Module A: Generator (Hybrid Ansatz)** - Generates high-quality candidate sequences using a mix of Symmetric Subspace seeds and Tensor Network-inspired ansatzes.
2.  **Module B: Classical Refiner (Memetic Tabu Search)** - A high-performance population-based solver with "Sprint" Tabu search for local polishing.
3.  **Module C: Q-Mobility Manager (Hatano-Nelson Drift)** - The "Physics Engine" that monitors stagnation. When the population converges, it triggers a **Non-Hermitian Shockwave** ($H_{HN}$) to forcibly delocalize elites and transport them to new, deeper basins.

## Theoretical Basis
See [Phase1_Theory.md](Phase1_Theory.md) for the mapping between Traffic Physics, Non-Hermitian Quantum Mechanics, and the LABS optimization landscape.

## Installation & Requirements

The code is designed to run on **NVIDIA GPUs** using `cupy` and `cudaq` (NVIDIA CUDA-Q).
It automatically detects hardware and falls back to CPU (`numpy`) if GPUs are unavailable.

Requirements:
- Python 3.8+
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `cupy` (Recommended for Phase 2 speed)
- `cudaq` (Optional, for Quantum simulation)

## How to Run

Run the main benchmark protocol:

```bash
python3 main.py
```

This will:
1.  Execute the **Scaling Benchmark** for $N=[30, 35, 40]$.
2.  Generate the **Phase 1 Robustness Plots** (`scaling_advantage.png`, `energy_shift_N40.png`, `ipr_vs_success.png`).
3.  Save the numerical results to `nvidia_challenge_results.csv`.

## File Structure

- `main.py`: Orchestrator, Benchmark Protocol, and Plotting Suite.
- `quantum_driver.py`: Implements the Non-Hermitian Hatano-Nelson Driver, Symplectic Integrator, and Dilation Circuit mapping.
- `tabu_search.py`: Implements the classical Tabu Search logic.
- `labs_utils.py`: GPU-accelerated (FFT-based) utility functions for $O(N \log N)$ energy calculation.
