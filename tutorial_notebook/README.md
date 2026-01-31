# MIT-iQuHACK-NVIDIA-Challenge: Non-Hermitian Counter-Diabatic Driving

This repository contains the solution for the NVIDIA Challenge at MIT iQuHACK.
The solution implements a **Hybrid Quantum-Classical Solver** for the Low Autocorrelation Binary Sequence (LABS) problem.

## Solution Architecture

The solver consists of three modules:
1.  **Module A: Generator (MPS)** - Generates candidate sequences using Tensor Network ansatz (simulated).
2.  **Module B: Classical Refiner (Tabu Search)** - Uses Memetic Tabu Search to locally optimize the Merit Factor.
3.  **Module C: Q-Mobility Manager (Hatano-Nelson Drift)** - Uses Non-Hermitian Physics to break detailed balance and escape local minima ("Traffic Congestion") when the solver stagnates.

## Theoretical Basis
See [Phase1_Theory.md](Phase1_Theory.md) for the mapping between Traffic Physics and the LABS optimization landscape.

## Installation & Requirements

The code is designed to run on **NVIDIA GPUs** using `cupy` and `cudaq` (NVIDIA CUDA-Q).
However, it includes a CPU fallback mode using `numpy` for development and testing.

Requirements:
- Python 3.8+
- `numpy`
- `scipy`
- `cupy` (Optional, for GPU acceleration)
- `cudaq` (Optional, for Quantum simulation)

## How to Run

Run the main solver script:

```bash
python3 main.py
```

This will:
1.  Initialize the solver for sequence length $N=40$ (configurable).
2.  Run the hybrid loop (Tabu Search + Shockwave).
3.  Output the best found Binary Sequence and its Merit Factor.

## File Structure

- `main.py`: Entry point and orchestration logic.
- `quantum_driver.py`: Implements the Non-Hermitian Hatano-Nelson Driver (Module C) and MPS Generator (Module A).
- `tabu_search.py`: Implements the classical Tabu Search (Module B).
- `labs_utils.py`: Utility functions for Merit Factor calculation (Phase 0).
