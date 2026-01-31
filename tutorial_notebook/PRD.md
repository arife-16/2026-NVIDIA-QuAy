# Product Requirements Document (PRD)

**Project Name:** Accelerated Quantum-Enhanced Memetic Search (A-QEMS)
**Team Name:** QuAy
**GitHub Repository:** https://github.com/arife-16/MIT-iQuHACK-NVIDIA-Challenge

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Arife Nur Ayaz | @arife-16 | @arife16 |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** **Gradient-Biased Non-Hermitian Optimization**
    * We have evolved the Hatano-Nelson model into a dynamic **Quantum Gradient Descent** driver.
    * **Gradient Mapping:** Unlike random drift, our drift parameter $\delta$ is determined by $\nabla E(s)$. This biases the quantum flow toward lower energy configurations, acting as a "quantum kick" through high-energy barriers.
    * **Symmetry-Protected Projection:** We implement an active projection operator $\mathcal{P}_{\mathbb{Z}_2 \times \mathbb{Z}_2}$ that maps the state back to the Symmetric ($S = S_{rev}$) or Skew-Symmetric ($S = -S_{rev}$) subspace after every integration step. This ensures that the quantum driver never wanders into redundant configurations, mathematically guaranteeing the 75% search space reduction.

* **Motivation:**
    * To solve the Low Autocorrelation Binary Sequence (LABS) problem for $N > 40$ by overcoming the "Glassy Landscape" limitation of standard quantum annealers.
    * We replace the tutorial's standard Counterdiabatic Driver with a **Gradient-Biased Non-Hermitian Driver**. This system utilizes energy-gradient-mapped imaginary potentials and a **First-Order Symplectic Split-Operator** scheme.
    * This architecture enables a **Digitized Counterdiabatic (DCQO)** approach that preserves the Hamiltonian structure during 'gradient kicks' while avoiding the numerical overhead of higher-order Runge-Kutta stages.
    * To implement the non-unitary Hatano-Nelson evolution on quantum hardware, we will utilize the Dilation Method, adding one ancilla qubit to embed the non-Hermitian operator into a larger unitary space, or Variational Imaginary Time Evolution (VITE) via CUDA-Q's optimization headers.

### Literature Review
* **Reference:** Hatano, N., & Nelson, D. R. (1996). "Localization Transitions in Non-Hermitian Quantum Mechanics." (*Phys. Rev. Lett*).
* **Reference:** Gomez Cadavid, A., et al. (2025). "Scaling advantage with quantum-enhanced memetic tabu search." (*arXiv:2511.04553*).
* **Reference:** Chandarana, P., et al. (2023). "Digitized counterdiabatic quantum algorithm for protein folding." (*Phys. Rev. Appl. 20, 014024*).

---

## 3. The Acceleration Strategy

### Quantum Acceleration (CUDA-Q)
* **Strategy:**
    * **Quantum Kernel:** We utilize CUDA-Q with the `tensornet` (Matrix Product State) backend. This allows us to simulate the non-unitary Hatano-Nelson evolution efficiently on GPU by compressing the state vector.
    * **Memory Management:** For $N > 40$ runs, we will constrain the tensornet Max Bond Dimension to $\chi = 64$ to prevent out-of-memory (OOM) errors on A100 instances while maintaining a fidelity threshold of $> 0.98$.

### Classical Acceleration (MTS)
* **Strategy:**
    * **Classical Kernel:** We replace the sequential Python loops of the Tabu Search with **CuPy**.
    * **Batching:** We evaluate the energy delta of all $N$ possible bit-flips simultaneously using GPU matrix operations ($O(1)$ parallel depth) rather than iteratively ($O(N)$).
    * **Algorithmic Optimization:** We replace the $O(N^2)$ scalar autocorrelation loop with an FFT-based autocorrelation (using `cupy.fft`), reducing complexity to $O(N \log N)$ for energy evaluations during the Memetic Search.

### Hardware Targets
* **Dev Environment:** L4 GPU (Tier 1) for Code debugging, Unit Tests, Small N ($N<30$) runs.
* **Production Environment:** A100 GPU (Tier 2) for "Hero Runs" ($N=40, 50, 60$) and final benchmarking.

---

## 4. The Verification Plan

### Unit Testing Strategy
* **Framework:** Automated test suite (`tests.py`) that runs before every major compute job.
* **AI Hallucination Guardrails:** We reject "ad-hoc" print statements. All logic is verified against physics invariants.

### Core Correctness Checks
* **Check 1 (Ground Truth Validation $N=3$):**
    * *Test:* Brute-force calculate all $2^3=8$ sequences.
    * *Assertion:* `calculate_energy([1, 1, -1]) == 1.0` (Manual verification).
* **Check 2 (Symmetry Invariant Testing):**
    * *Test:* Generate random sequence $S$.
    * *Assertion:* $\text{Energy}(S) == \text{Energy}(-S)$ **AND** $\text{Energy}(S) == \text{Energy}(\text{Reverse}(S))$.
    * *Purpose:* Ensures the Hamiltonian construction hasn't violated physical symmetries.
* **Check 3 (Benchmark Calibration Barker-13):**
    * *Test:* Input the known Barker-13 sequence.
    * *Assertion:* $\text{Merit\_Factor}(\text{Barker13}) \approx 14.08$.
    * *Purpose:* Calibrates the objective function against known literature values.
* **Check 4 (Quantum Sanity Check):**
    * *Test:* Compare Mean Energy of Quantum Ansatz ($E_Q$) vs. Random Noise ($E_R$) for $N=20$.
    * *Assertion:* $E_Q < E_R$ (The quantum algorithm must provide a head-start).
* **Check 5 (Symplectic Invariant Assertion):**
    * *Test:* Run the evolution for 100 Trotter steps with $\Delta t = 0.01$.
    * *Assertion:* Energy deviation $\Delta E < 10^{-5}$ across the trajectory.
    * *Purpose:* Validates that the **Velocity Verlet** / Split-Operator integrator preserves the phase-space volume, essential for long-range sequence optimization.
* **Check 6 (Stagnation Recovery Test):**
    * *Test:* Apply Gradient-Biased Shockwave to a trapped population (e.g., $E=108$).
    * *Assertion:* Post-shock mean Inverse Participation Ratio (IPR) $< 0.5$.
    * *Purpose:* Confirms the driver successfully delocalizes the state for tunneling out of local minima.

---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:**
    * **Agent 1 (The Architect):** Responsible for mapping the Physics equations to the LABS Hamiltonian. Gemini Pro will be utilized for the scientific ground.
    * **Agent 2 (The Coder):** Responsible for translating the math into `cupy` kernels and refactoring the `main.py` loop. TRAE SOLO agent will be used for this task.

### Success Metrics
* **Metric 1 (Mathematical Stability):** Zero divergence in the energy gradient during $N=40$ symplectic 'kicks', ensuring convergence to $E \le 108$ basins within the Tier 2 budget.
* **Metric 2 (Optimality):** Our target for $N=40$ is a Merit Factor $F > 8.0$. We specifically target the escape of the $E=108$ local minimum by asserting a post-shockwave IPR $< 0.5$, ensuring the population has tunneled into the global search space.
* **Metric 3 (Efficiency):** Achieve a 75% reduction in redundant state evaluations through Symmetry Protection.

### Visualization Plan
* **Plot 1:** Time-to-Solution Scaling ($O(1.24^N)$ vs $O(1.34^N)$).
* **Plot 2:** Energy Distribution Shift (Violin Plot) showing Phase-Space Compression.
* **Plot 3:** Success Probability vs. IPR (Tunneling Evidence).

---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC

* **Plan:** We treat compute credits as cash. We utilize a Tiered Deployment Strategy:

| Tier | Hardware | Cost | Usage Strategy | Est. Cost |
| :--- | :--- | :--- | :--- | :--- |
| Tier 1: Dev | L4 GPU | $\sim \$0.80$/hr | Code debugging, Unit Tests, Small N ($N<30$) runs. | 5 hrs = \$4.00 |
| Tier 2: Prod | A100 GPU | $\sim \$2.50$/hr | Only for "Hero Runs" ($N=40, 50, 60$) and final benchmarking. | 4 hrs = \$10.00 |
| Buffer | N/A | N/A | Reserved for mistakes/overruns. | \$6.00 |

* **Protocol:** Auto-shutdown script will be implemented on all batch jobs to ensure no instance runs overnight.
