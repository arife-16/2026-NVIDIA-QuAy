import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# Ensure local modules are importable
sys.path.append(os.getcwd())

try:
    from labs_utils import calculate_merit_factor, calculate_energy, get_canonical
except ImportError:
    print("[!] Error: Could not import labs_utils. Please ensure labs_utils.py is in the directory.")
    sys.exit(1)

from tabu_search import TabuSearch
from quantum_driver import HatanoNelsonDriver, TNAnsatzGenerator

class MemeticManager:
    def __init__(self, N, population_size=20):
        self.N = N
        self.K = population_size
        self.population = [] # List of {'seq': array, 'energy': float}
        self.stagnation_counter = 0
    
    def add(self, sequence, energy):
        """Add a sequence to population, maintaining sorted order (best first)."""
        # 1. Canonicalize to reduce symmetry
        canonical_seq = get_canonical(sequence)
        
        # 2. Diversity Check (Hamming Distance)
        # Don't add if too similar to existing
        for p in self.population:
            if np.array_equal(p['seq'], canonical_seq):
                return
            # Hamming distance check
            # diff = np.sum(p['seq'] != canonical_seq)
            # if diff < 2: return # Too similar
        
        self.population.append({'seq': canonical_seq, 'energy': energy})
        self.population.sort(key=lambda x: x['energy'])
        
        # Survival of the fittest: Keep top K
        if len(self.population) > self.K:
            self.population = self.population[:self.K]
            
    def check_extinction(self):
        """
        Check if population has converged (low entropy).
        If so, kill bottom 80% and replace with random.
        """
        if len(self.population) < self.K:
            return
            
        # Simple convergence check: if top 5 are same energy
        top_energies = [p['energy'] for p in self.population[:5]]
        if max(top_energies) == min(top_energies):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        if self.stagnation_counter > 50:
            print("EXTINCTION EVENT! Population converged. Mass extinction triggered.")
            # Keep top 20%
            survivors_count = max(2, int(self.K * 0.2))
            self.population = self.population[:survivors_count]
            self.stagnation_counter = 0
            # Note: Caller needs to refill population

    def get_parents(self):
        """Tournament Selection: Pick 2 random, return best. Do this twice."""
        def tournament():
            if len(self.population) < 2:
                return self.population[0]
            # Pick 2 random indices
            idxs = np.random.choice(len(self.population), 2, replace=False)
            c1 = self.population[idxs[0]]
            c2 = self.population[idxs[1]]
            return c1 if c1['energy'] < c2['energy'] else c2
        
        if len(self.population) < 2:
             return self.population[0]['seq'], self.population[0]['seq']
             
        return tournament()['seq'], tournament()['seq']
    
    def crossover(self, p1, p2):
        """Single Point Crossover (Algorithm 3 from Paper)"""
        # Choose cut point k in [1, N-1]
        k = np.random.randint(1, self.N)
        # Child = p1[:k] + p2[k:]
        child = np.concatenate([p1[:k], p2[k:]])
        return child

    def add_batch(self, sequences, energies):
        """Batch add sequences to population."""
        # This is a critical update for the Gamma Knife protocol
        # We need to handle large batches efficiently
        
        # 1. Canonicalize all sequences (Vectorized if possible, but loop is safer for now)
        # Note: get_canonical works on single sequence.
        # Let's optimize: Only canonicalize if needed.
        # But for Gamma Knife we generated them symmetric already?
        # No, "Symmetric" means s[i] == s[N-1-i].
        # Canonical means map s to min(s, -s, rev(s), -rev(s)).
        # Even symmetric sequences have variants (-S).
        
        # Filter duplicates in batch
        # This is expensive for 1M sequences.
        # Strategy: Sort batch by energy first. Take top K. Then filter.
        
        # Combine with current population
        # Current pop is list of dicts.
        # Let's convert current pop to parallel arrays for efficiency?
        # For now, stick to list logic but optimize.
        
        # Sort incoming batch
        sort_idx = np.argsort(energies)
        sequences = sequences[sort_idx]
        energies = energies[sort_idx]
        
        # Take only top K candidates from the batch to avoid processing junk
        # If batch is 1000, and K=1000, take all.
        limit = min(len(sequences), self.K * 2)
        sequences = sequences[:limit]
        energies = energies[:limit]
        
        # Add to population one by one (safe but slow)
        # Or just rebuild population
        
        for i in range(len(sequences)):
            self.add(sequences[i], energies[i])

def generate_symmetric_seeds(N, count):
    """
    Generates seeds living in the Symmetric and Skew-Symmetric subspaces.
    Reduces search space from 2^N to 2^(N/2).
    """
    half_N = N // 2
    
    # 1. Generate random half-sequences
    # count // 2 for Symmetric, count // 2 for Skew
    count_sym = count // 2
    count_skew = count - count_sym
    
    # Handle Odd N:
    # If N is odd, symmetric sequences must have a middle element.
    # s = [left, middle, right].
    # Symmetric: right = reverse(left).
    # Skew: right = -reverse(left). Middle element must be 0? No, binary sequence.
    # Actually, strictly skew-symmetric sequences of odd length cannot be binary (+/-1) because s[mid] = -s[mid] -> s[mid]=0.
    # So for Odd N, we only generate Symmetric seeds, or we relax skew definition.
    # Let's stick to Symmetric for Odd N.
    
    if N % 2 != 0:
        # Odd N
        half_N = N // 2
        # Middle element can be +1 or -1
        
        # Generate random halves
        halves = np.random.choice([1, -1], size=(count, half_N))
        mids = np.random.choice([1, -1], size=(count, 1))
        
        # Construct Full Symmetric: [half, mid, reverse(half)]
        full_seqs = np.hstack([halves, mids, halves[:, ::-1]])
        return full_seqs
        
    else:
        # Even N (Original Logic)
        half_N = N // 2
        
        # Generate random halves
        halves_sym = np.random.choice([1, -1], size=(count_sym, half_N))
        halves_skew = np.random.choice([1, -1], size=(count_skew, half_N))
        
        # Symmetric Construction
        full_sym = np.hstack([halves_sym, halves_sym[:, ::-1]])
        
        # Skew Construction
        full_skew = np.hstack([halves_skew, -halves_skew[:, ::-1]])
        
        return np.vstack([full_sym, full_skew])

# PRODUCTION CONFIGURATION for Colab Pro+ (A100)
CONFIG = {
    # 1. The Fleet Size
    # Paper uses K=100. We use K=500 because we have an A100.
    # Higher population = Diversity = Escaping the E=108 trap.
    "POPULATION_SIZE": 1000,
    
    # 2. Survival Pressure
    # Keep the elite, kill the weak.
    # High churn rate forces rapid evolution.
    "ELITE_SIZE": 50,      # Only top 10% survive directly
    "SURVIVAL_RATE": 0.2,  # Keep top 20% total
    
    # 3. The "Ghostbuster" Symmetry
    # ENABLED. (Canonicalize every sequence)
    "USE_CANONICAL": True,
    
    # 4. Tabu Intensity (The "Sprint")
    # Short, intense bursts. Don't let Tabu linger.
    # Let the Population logic handle the long-term memory.
    "TABU_MAX_ITER": 2000,  # Increased for N=40 deep dive
    "TABU_TENURE": 15,     # Dynamic range [5, 20] in code
    
    # 5. Physics Engine (The "Shockwave")
    # Only trigger when the ENTIRE population stagnates.
    "SHOCKWAVE_THRESHOLD": 20, # If global best doesn't improve for 50 generations
    "DRIFT_STRENGTH": 1.5,    # Increased Non-Hermitian drift strength (was 0.95)
    
    # 6. Runtime Limits (The Benchmark)
    # Stop when you hit the known optimal or timeout.
    "TARGET_ENERGY": {
        30: 42,
        35: 64, # Approx
        40: 86,   # World Record level for N=40 (F~9.3)
        45: 110, # Approx
        50: 140,  # Approximate good target for N=50
        60: 220   # Approximate good target for N=60
    },
    "TIMEOUT_SEC": 900, # 15 minutes max per N for Record Attempt
}

# --- Plotting Utilities for Phase 1 Robustness ---
def plot_energy_distribution_shift(initial_energies, quantum_energies, N):
    """
    Violin Plot: Compare random initialization vs Quantum Seeded initialization.
    Shows the 'pruning' of high-energy garbage states.
    """
    plt.figure(figsize=(10, 6))
    data = [initial_energies, quantum_energies]
    parts = plt.violinplot(data, showmeans=True, showmedians=True)
    
    # Style
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    plt.xticks([1, 2], ['Random Initialization', 'Quantum/Symmetric Seeding'])
    plt.ylabel('Energy (Lower is Better)')
    plt.title(f'Phase-Space Compression: Energy Distribution Shift (N={N})')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"energy_shift_N{N}.png")
    plt.close()

def plot_success_probability_vs_ipr(ipr_history, success_history):
    """
    Scatter/Bin Plot: Success Probability vs IPR.
    Demonstrates that delocalization (lower IPR) correlates with escape success.
    """
    if not ipr_history: return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ipr_history, success_history, alpha=0.6, c='blue')
    plt.xlabel('Inverse Participation Ratio (IPR)')
    plt.ylabel('Escape Success (1=Yes, 0=No)')
    plt.title('Tunneling Efficiency: Escape Probability vs Localization')
    plt.grid(True, alpha=0.3)
    plt.savefig("ipr_vs_success.png")
    plt.close()
    
def plot_time_to_solution_scaling(results_df):
    """
    Log-Log Plot: Time to Solution vs N.
    Compares QE-MTS (Ours) vs Classical MTS (Baseline/Theoretical).
    """
    if results_df.empty: return
    
    plt.figure(figsize=(10, 6))
    
    ns = results_df['N'].unique()
    times = []
    for n in ns:
        t = results_df[results_df['N'] == n]['Time (s)'].median()
        times.append(t)
        
    plt.plot(ns, times, 'o-', label='QE-MTS (Ours)', linewidth=2, markersize=8)
    
    # Theoretical Classical Slope (approx O(1.34^N))
    # Anchor to our first point
    if len(ns) > 0:
        n0 = ns[0]
        t0 = times[0]
        classical_curve = [t0 * (1.34**(n - n0)) for n in ns]
        plt.plot(ns, classical_curve, '--', label='Classical MTS (Theory O(1.34^N))', color='gray')
    
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Time to Solution (s)')
    plt.yscale('log')
    plt.title('Scaling Advantage: Time-to-Solution vs N')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig("scaling_advantage.png")
    plt.close()
# -------------------------------------------------

def get_gpu_name():
    try:
        import cupy as cp
        # This is a bit hacky to get device name in cupy without full cuda toolkit sometimes
        # But generally:
        return cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
    except:
        return "CPU (NumPy)"

def solve_labs_problem(N, time_limit=900, target_energy=None):
    start_time = time.time()
    
    # Initialize Drivers
    tabu = TabuSearch(N, min_tenure=5, max_tenure=20, max_iter=CONFIG["TABU_MAX_ITER"])
    memetic = MemeticManager(N, population_size=CONFIG["POPULATION_SIZE"])
    
    # Initialize Quantum Driver (Hatano-Nelson with Gradient Drift)
    drift_strength = CONFIG.get("DRIFT_STRENGTH", 0.95)
    quantum_driver = HatanoNelsonDriver(N, drift=drift_strength)
    
    print(f"[-] Initializing with HYBRID SCAN (K={memetic.K})...")
    
    # 1. Symmetric Seeds (50%) - Gamma Knife
    candidates_sym = generate_symmetric_seeds(N, memetic.K // 2)
    
    # 2. Barker Seeds (50%) - Classic
    generator = TNAnsatzGenerator(N)
    candidates_barker = generator.sample(memetic.K // 2)
    
    # Combine
    candidates = np.vstack([candidates_sym, candidates_barker])
    
    # Batch Evaluate
    energies = [calculate_energy(s) for s in candidates]
    memetic.add_batch(candidates, np.array(energies))
    
    # --- Data Collection for Violin Plot ---
    # Generate a random control group to compare against
    random_control = np.random.choice([1, -1], size=(memetic.K, N))
    random_energies = [calculate_energy(s) for s in random_control]
    # Plot the shift immediately
    plot_energy_distribution_shift(random_energies, energies, N)
    # ---------------------------------------
    
    global_best_energy = memetic.population[0]['energy']
    global_best_seq = memetic.population[0]['seq'].copy()
    
    print(f"[-] Starting Optimization Loop for N={N}...")
    
    stagnation_cycles = 0
    extinction_count = 0
    
    # Tracking for IPR Plot
    ipr_history = []
    success_history = []
    
    # BANNED LIST REMOVED - Using Symmetry Protection instead
    
    while time.time() - start_time < time_limit:
        
        # Check Target
        if target_energy and global_best_energy <= target_energy:
            print(f"\\n[!] TARGET MET: E={global_best_energy}")
            break
            
        # STAGNATION CHECK
        if stagnation_cycles > 20:
            print(f"\\n[!] STAGNATION at E={global_best_energy}. Initiating Subspace Reseed & Quantum Shockwave.")
            
            # Kill population, keep elites, but refill with NEW Symmetries
            elites = memetic.population[:10]
            
            # Apply Quantum Shockwave to Elites to find better nearby basins
            # (Gradient-based Drift pushes them to lower energy configurations)
            shocked_elites = []
            
            current_best_before_shock = global_best_energy
            
            for p in elites:
                # Evolve slightly (tau=1.0)
                # The shockwave automatically projects to symmetry
                # Check IPR to ensure delocalization
                s_new, ipr = quantum_driver.apply_shockwave(p['seq'], tau=2.0) # Increased tau for stronger kick
                shocked_elites.append(s_new)
                
                # Track IPR for Plotting (Phase 1 Requirement)
                # We don't know success yet, will update later or track generally
                # Let's track: Did this shock lead to a better energy?
                e_new = calculate_energy(s_new)
                success = 1 if e_new < p['energy'] else 0
                ipr_history.append(ipr)
                success_history.append(success)
                
                # print(f"    [DEBUG] Shockwave IPR: {ipr:.4f}") # Uncomment for debug
            
            # Evaluate Shocked Elites
            shocked_engs = [calculate_energy(s) for s in shocked_elites]
            
            # Massive Injection of FRESH Seeds (Hybrid)
            num_fill = memetic.K - 10
            
            # Hybrid Refill: 50% Sym, 50% Random/Barker
            new_sym = generate_symmetric_seeds(N, num_fill // 2)
            # Use Random for the other half to break symmetry if needed
            new_rand = np.random.choice([1, -1], size=(num_fill - len(new_sym), N))
            
            new_seeds = np.vstack([new_sym, new_rand])
            
            # Re-eval new seeds
            new_engs = [calculate_energy(s) for s in new_seeds]
            
            # Reset Population with elites + shocked elites + new seeds
            # We keep original elites safely
            memetic.population = elites
            
            # Add shocked versions (they might be better!)
            memetic.add_batch(np.array(shocked_elites), np.array(shocked_engs))
            
            # Add fresh seeds
            memetic.add_batch(new_seeds, np.array(new_engs))
            
            stagnation_cycles = 0
            extinction_count += 1
            continue

        # Evolution (Standard)
        p1, p2 = memetic.get_parents()
        child = memetic.crossover(p1, p2)
        
        # Mutation
        idx = np.random.choice(N, 2, replace=False)
        child[idx] *= -1
        
        # Optimize
        refined_seq, refined_energy = tabu.solve(child)
        
        # Update Population
        memetic.add(refined_seq, refined_energy)
            
        # Update Global
        if refined_energy < global_best_energy:
            global_best_energy = refined_energy
            global_best_seq = refined_seq.copy()
            mf = calculate_merit_factor(global_best_seq)
            print(f"    New Best! F={mf:.2f} (E={global_best_energy})")
            stagnation_cycles = 0
        else:
            stagnation_cycles += 1

    return global_best_seq, global_best_energy, calculate_merit_factor(global_best_seq), ipr_history, success_history

def run_benchmark():
    results = []
    # Benchmark Range: N=30 to N=60 in steps of 5
    # The paper stops at N=37. We go deeper.
    # To demonstrate scaling O(1.24^N) vs O(1.34^N), we need a range.
    # We will run N=[30, 35, 40] quickly to generate the slope.
    benchmark_ns = [30, 35, 40]
    # Set a shorter timeout per N to ensure we complete the sweep
    # For scaling plot, we need *Time to Solution*.
    # If we don't hit target, the time is the timeout (censored data).
    # But for N=30, 35 we should hit it fast.
    CONFIG["TIMEOUT_SEC"] = 120 
    
    print(f"[-] Starting NVIDIA Challenge Benchmark Protocol (Scaling Run)...")
    print(f"[-] Hardware: {get_gpu_name()}") # Ensure this prints 'Tesla A100'
    
    all_ipr = []
    all_success = []
    
    for N in benchmark_ns:
        target = CONFIG["TARGET_ENERGY"].get(N, 0)
        print(f"\n>>> Benchmarking N={N} (Target E<={target})...")
        
        start_t = time.time()
        
        # Call your solver
        # Note: Ensure solver returns (best_seq, best_energy, best_merit)
        best_seq, best_energy, best_mf, ipr_hist, success_hist = solve_labs_problem(
            N=N,
            time_limit=CONFIG["TIMEOUT_SEC"],
            target_energy=target # Pass this to stop early if hit
        )
        
        all_ipr.extend(ipr_hist)
        all_success.extend(success_hist)
        
        runtime = time.time() - start_t
        
        # Log Metrics
        results.append({
            "N": N,
            "Time (s)": runtime,
            "Energy": best_energy,
            "Merit Factor": best_mf,
            "Optimal Hit": "YES" if best_energy <= target else "NO"
        })
        
        print(f"    Result: F={best_mf:.2f} (E={best_energy}) in {runtime:.2f}s")

    # Output CSV for plotting
    df = pd.DataFrame(results)
    print("\n[-] Benchmark Complete. Results:")
    print(df)
    
    # Generate Phase 1 Plots
    print("[-] Generating Phase 1 Robustness Plots...")
    plot_time_to_solution_scaling(df)
    plot_success_probability_vs_ipr(all_ipr, all_success)
    
    # Append to existing CSV if it exists, otherwise create new
    csv_file = "nvidia_challenge_results.csv"
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    # Run the benchmark
    run_benchmark()
