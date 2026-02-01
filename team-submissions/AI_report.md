# AI Report: Orchestrating Quantum-Classical Hybrids
## MIT iQuHACK NVIDIA Challenge 2025

---

## 1. The Workflow

We utilized the **Trae IDE** agent as a fully autonomous "Senior Pair-Programmer". Instead of treating the AI as a simple code generator, we integrated it into an iterative OODA loop (Observe-Orient-Decide-Act):

*   **Coding Agent (Trae)**: Handled implementation of complex quantum kernels (`quantum_driver.py`) and refactoring of physics engines.
*   **Diagnostic Orchestration**: We tasked the agent with creating *self-verification tools* (`debug_analysis.py`) rather than just unit tests, effectively asking the AI to "debug itself."
*   **Documentation**: The same context-aware agent generated our final presentation and this report, ensuring technical accuracy based on the actual code state.

---

## 2. Verification Strategy

We adopted a **"Diagnostic-First"** verification strategy. Since quantum simulation backends can behave unpredictably regarding qubit ordering and register management, standard unit tests were insufficient.

### Specific Verification: `debug_analysis.py`
To catch AI hallucinations regarding backend behavior, we prompted the agent to write a `probe_kernel` test:

1.  **Hallucination Caught (Physics)**: The AI initially implemented a "Symplectic Integrator" for the classical fallback, assuming Hamiltonian dynamics would solve the optimization problem.
    *   **The Catch**: Our diagnostic script traced the energy evolution and found it *increasing* (24.0 → 25.2), proving the AI's physics model was energy-conserving rather than energy-minimizing.
    *   **The Fix**: We forced a pivot to **Damped Gradient Descent**, which the diagnostic confirmed immediately (Energy 24.0 → 14.7).

2.  **Hallucination Caught (Backend)**: The AI assumed `mz(ancilla)` and `mz(qubits)` would return concatenated results.
    *   **The Catch**: The diagnostic revealed that for $N=8$, the returned bitstrings had length 8 instead of 9, meaning the ancilla measurement was silently dropped.
    *   **The Fix**: We refactored the kernel to use a **Unified Register (`qvector(N+1)`)**, guaranteeing atomic measurement of the full system.

---

## 3. The "Vibe" Log

### 🏆 Win
**The LSB Discovery**: The AI saved us hours of debugging by writing a script that empirically tested bit ordering.
*   *Scenario*: We were getting 0.00% success rate. We didn't know if the Ancilla was MSB, LSB, or missing.
*   *AI Action*: It wrote a `probe_kernel` that initialized Ancilla to $|1\rangle$ and System to $|0\rangle$.
*   *Result*: The output `00001` proved definitively that Ancilla was at the LSB (Index -1), allowing us to fix our post-selection logic immediately.

### 🧠 Learn
**Physics over Syntax**: We improved our results significantly when we stopped prompting for "syntax fixes" and started prompting for **"physics understanding"**.
*   *Shift*: Instead of "Fix the Unsupported Type error", we asked "Lets understand what is wrong with the physics of the codebase".
*   *Outcome*: The AI analyzed the Hamiltonian dynamics, realized the lack of dissipation/friction was the root cause of the optimization failure, and rewrote the solver to use Pseudo-Langevin dynamics.

### 💀 Fail
**The "Missing Bit" Hallucination**:
*   *Failure*: The AI repeatedly insisted that `mz(ancilla)` followed by `mz(qubits)` would work, even as logs showed bitstrings of length $N$ instead of $N+1$. It assumed the backend behaves like a standard simulator where registers are always preserved.
*   *Fix*: We had to explicitly prompt the AI to "Analyze DEBUG bitstrings to determine correct Ancilla position", forcing it to look at the *evidence* (length 8 strings) rather than its internal model of how CUDA-Q *should* work. This led to the Unified Register refactor.

---

## 4. Context Dump

### Key Diagnostic Prompt
> "to solve the issue, give me a comprehensive analysis script to run to spot what is going wrong"

This simple prompt generated `debug_analysis.py`, which became our primary tool for solving the challenge.

### The Physics Correction Prompt
> "lets understand what is wrong with the physics of the codebase, the quantum script should run, and even if we fall back to classic, the energy minimization should be working correctly"

This prompt triggered the shift from broken Symplectic dynamics to working Gradient Descent.

### Artifact: `debug_analysis.py` (Snippet)
```python
def test_quantum_kernel():
    # ...
    print("[1] Bit Ordering & Ancilla Position Check")
    # Probe kernel to find where the Ancilla is hiding
    result = cudaq.sample(probe_kernel, N_probe, shots_count=100)
    # ...
    if first_key[0] == '1': print("-> Ancilla appears at MSB")
    elif first_key[-1] == '1': print("-> Ancilla appears at LSB")
```
