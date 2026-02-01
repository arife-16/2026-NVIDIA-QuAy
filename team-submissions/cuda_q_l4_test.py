#!/usr/bin/env python3
"""
CUDA-Q L4 GPU Test Script

This script is optimized for testing the quantum kernel on Brev L4 GPU instances.
L4 is Tier 1 hardware - great for development and testing.

Key differences from A100:
- L4: 24GB VRAM, good for N up to ~30
- A100: 40/80GB VRAM, can handle N up to ~50+
- L4: Lower cost (~$0.80/hr vs $2.50/hr)
- L4: Perfect for debugging and initial validation

Usage on Brev L4:
    python3 cuda_q_l4_test.py [--quick]
"""

import numpy as np
import sys
import os
import time
import json
from datetime import datetime

# Ensure local modules are importable
sys.path.append(os.getcwd())

from quantum_driver import HatanoNelsonDriver, HAS_CUDAQ
from labs_utils import calculate_energy, get_canonical
import multiprocessing

def run_quantum_worker(N, sequence, bond_dim, shots, queue):
    """Worker process for quantum execution to allow hard timeout."""
    try:
        import cudaq
        from quantum_driver import HatanoNelsonDriver
        
        # Configure backend in this process
        cudaq.set_target('tensornet', max_bond_dimension=bond_dim)
        
        # Initialize driver
        driver = HatanoNelsonDriver(N, t=1.0, drift=0.5)
        
        # Run evolution
        evolved_state, ancilla_stats = driver._evolve_quantum_dcqo(
            sequence.astype(float), tau=0.2, track_ancilla_stats=True
        )
        
        queue.put(('success', evolved_state, ancilla_stats))
        
    except Exception as e:
        queue.put(('error', str(e)))

def l4_environment_check():
    """Check if we're running on L4 GPU environment."""
    print("L4 ENVIRONMENT CHECK")
    print("="*50)
    
    # Check CUDA availability
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"GPU: {gpu_info}")
            
            # Check if it's actually L4
            if 'L4' in gpu_info:
                print("✓ Confirmed L4 GPU")
            elif 'A100' in gpu_info:
                print("⚠️  This is an A100, not L4 (you can use higher N values)")
            else:
                print(f"⚠️  GPU detected but not L4/A100: {gpu_info}")
        else:
            print("GPU: nvidia-smi failed")
    except Exception as e:
        print(f"GPU: Could not detect ({e})")
    
    # Check memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"System Memory: {mem.total // (1024**3)} GB")
    except:
        print("System Memory: Unknown")
    
    # Check CUDA-Q
    if HAS_CUDAQ:
        try:
            import cudaq
            print(f"CUDA-Q Version: {cudaq.__version__}")
            
            # Test backend availability
            backends = ['tensornet', 'nvidia', 'default']
            available_backends = []
            
            for backend in backends:
                try:
                    cudaq.set_target(backend)
                    available_backends.append(backend)
                except:
                    pass
            
            print(f"Available Backends: {available_backends}")
            
            # L4-specific recommendations
            if available_backends:
                print("\nL4 Optimization Tips:")
                print("  • Use max_bond_dimension=32 for N≤20")
                print("  • Use max_bond_dimension=64 for N≤30")
                print("  • Avoid N>30 on L4 (use A100 for larger)")
            
            return available_backends
            
        except Exception as e:
            print(f"CUDA-Q Error: {e}")
            return []
    else:
        print("CUDA-Q: Not available")
        return []

def test_quantum_kernel_l4(n_values=[8, 12, 16, 20, 24], quick_mode=False):
    """
    L4-optimized test of quantum kernel.
    
    Args:
        n_values: System sizes to test (default optimized for L4)
        quick_mode: If True, use fewer shots and iterations for faster testing
    """
    print("\nQUANTUM KERNEL L4 TEST")
    print("="*60)
    
    if not HAS_CUDAQ:
        print("❌ CUDA-Q not available")
        return None
    
    import cudaq
    
    # L4-optimized configuration
    backend = 'tensornet'
    
    # Adaptive bond dimension based on N
    def get_bond_dim(N):
        if N <= 16:
            return 32
        elif N <= 24:
            return 48
        else:
            return 64
    
    # Quick mode for rapid iteration
    shots = 1024 if quick_mode else 2048
    
    print(f"Backend: {backend}")
    print(f"Shots: {shots}")
    print(f"Quick mode: {quick_mode}")
    print(f"Testing N values: {n_values}")
    
    results = {}
    
    for N in n_values:
        print(f"\n{'='*40}")
        print(f"Testing N={N}")
        print(f"{'='*40}")
        
        bond_dim = get_bond_dim(N)
        print(f"Bond dimension for N={N}: {bond_dim}")
        
        try:
            # Configure backend with adaptive bond dimension
            cudaq.set_target(backend, max_bond_dimension=bond_dim)
            
            # Initialize driver
            driver = HatanoNelsonDriver(N, t=1.0, drift=0.5)
            
            # Test sequence
            test_sequence = np.random.choice([1, -1], size=N)
            initial_energy = calculate_energy(test_sequence)
            
            print(f"Initial sequence: {test_sequence}")
            print(f"Initial energy: {initial_energy:.6f}")
            
            # Test quantum evolution with ancilla tracking
            start_time = time.time()
            
            # Use Multiprocessing for Hard Timeout (Signal doesn't interrupt C++)
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=run_quantum_worker, 
                args=(N, test_sequence, bond_dim, shots, queue)
            )
            p.start()
            p.join(timeout=300)
            
            if p.is_alive():
                print(f"❌ N={N} Timed out (>300s) - Killing process")
                p.terminate()
                p.join()
                
                # Record timeout result
                results[N] = {
                    'bond_dimension': bond_dim,
                    'evolution_time': 300.0,
                    'final_energy': initial_energy,
                    'energy_improvement': 0.0,
                    'ancilla_success_rate': 0.0,
                    'method_used': 'timeout',
                    'classical_time': 300.0,
                    'classical_energy': initial_energy,
                    'classical_attempts': 0,
                    'speedup': 0.0,
                    'energy_ratio': 1.0,
                    'classical_ipr': 0.0
                }
                try:
                    if 'generate_plots' in globals():
                        generate_plots(results, 'benchmark_incremental')
                except: pass
                continue
            else:
                # Process finished
                if not queue.empty():
                    status, *data = queue.get()
                    if status == 'success':
                        evolved_state, ancilla_stats = data
                    else:
                        raise RuntimeError(f"Worker failed: {data[0]}")
                else:
                    raise RuntimeError("Worker finished but returned no data")

            evolution_time = time.time() - start_time
            
            # Results
            final_energy = calculate_energy(evolved_state)
            ancilla_success_rate = ancilla_stats['success_rate']
            total_shots = ancilla_stats.get('total_shots', shots)
            method_used = ancilla_stats.get('method', 'unknown')
            
            print(f"Evolution time: {evolution_time:.3f}s")
            print(f"Final energy: {final_energy:.6f}")
            print(f"Energy improvement: {initial_energy - final_energy:.6f}")
            print(f"Ancilla success rate: {ancilla_success_rate:.3f} ({ancilla_success_rate*100:.1f}%)")
            print(f"Method used: {method_used}")
            
            # Compare with classical - TIME MATCHED
            print(f"\nClassical comparison (Time Budget: {evolution_time:.3f}s)...")
            classical_start = time.time()
            best_classical_energy = float('inf')
            classical_attempts = 0
            classical_ipr = 0.0
            
            # Run classical solver until time budget exceeded (Multi-Start Local Search)
            # This ensures fair comparison by giving Classical same CPU time as Quantum used on GPU
            while True:
                # Use random restart or perturbation
                if classical_attempts == 0:
                    current_seq = test_sequence.copy()
                else:
                    # Random restart
                    current_seq = np.random.choice([1, -1], size=N)
                
                # Run one classical descent
                c_seq, c_ipr = driver.apply_shockwave(current_seq, tau=0.2, use_quantum=False)
                c_energy = calculate_energy(c_seq)
                
                if c_energy < best_classical_energy:
                    best_classical_energy = c_energy
                    classical_ipr = c_ipr # Track IPR of best
                
                classical_attempts += 1
                
                elapsed = time.time() - classical_start
                if elapsed >= evolution_time:
                    break
                    
                # Avoid infinite loop if quantum was instant (e.g. < 0.01s)
                if evolution_time < 0.01 and classical_attempts > 0:
                    break

            classical_time = time.time() - classical_start
            print(f"Classical best energy: {best_classical_energy:.6f} (Attempts: {classical_attempts})")
            
            # Performance metrics
            # Speedup is now 1.0x by definition (Time Matched)
            # We compare ENERGY QUALITY
            energy_ratio = final_energy / best_classical_energy if best_classical_energy != 0 else 1
            classical_energy = best_classical_energy # Update for storage
            speedup = classical_time / evolution_time if evolution_time > 0 else 0
            
            print(f"\nPerformance:")
            print(f"  Time ratio (classical/quantum): {speedup:.2f}x")
            print(f"  Energy ratio (quantum/classical): {energy_ratio:.4f}")
            
            # Store results
            results[N] = {
                'bond_dimension': bond_dim,
                'evolution_time': evolution_time,
                'final_energy': final_energy,
                'energy_improvement': initial_energy - final_energy,
                'ancilla_success_rate': ancilla_success_rate,
                'method_used': method_used,
                'classical_time': classical_time,
                'classical_energy': classical_energy,
                'classical_attempts': classical_attempts,
                'speedup': speedup,
                'energy_ratio': energy_ratio,
                'classical_ipr': classical_ipr
            }
            
            # Incremental plotting (overwrite to keep latest)
            try:
                # We need to access generate_plots which is defined later.
                # Use a try-except to avoid crashing if not yet defined or imported.
                if 'generate_plots' in globals():
                    generate_plots(results, 'benchmark_incremental')
            except Exception:
                pass # Fail silently during loop
            
            # Status indicators
            status_msg = []
            if method_used == 'quantum':
                status_msg.append("✓ Quantum kernel successful")
            elif 'classical' in method_used:
                status_msg.append("⚠️  Used classical fallback")
            
            if ancilla_success_rate < 0.001:
                status_msg.append("⚠️  Very low ancilla success rate")
            elif ancilla_success_rate < 0.05:
                status_msg.append("⚠️  Low ancilla success rate")
            
            if energy_ratio < 0.95:
                status_msg.append("✓ Quantum achieved better energy")
            elif energy_ratio > 1.05:
                status_msg.append("⚠️  Classical achieved better energy")
            
            for msg in status_msg:
                print(msg)
            
            print(f"\n✓ Test completed for N={N}")
            
        except Exception as e:
            print(f"❌ Failed for N={N}: {e}")
            import traceback
            traceback.print_exc()
            results[N] = None
    
    return results

def print_l4_summary(results):
    """Print L4-specific benchmark summary."""
    if not results:
        return
    
    print("\n" + "="*60)
    print("L4 BENCHMARK SUMMARY")
    print("="*60)
    
    # Results table
    print(f"\n{'N':<6} {'Bond':<6} {'Method':<12} {'Energy':<10} {'Time(s)':<10} {'Success%':<10}")
    print("-"*60)
    
    for N, data in results.items():
        if data is None:
            print(f"{N:<6} {'FAILED':<6}")
            continue
        
        bond = data['bond_dimension']
        method = data['method_used'][:11]  # Truncate if needed
        energy = data['final_energy']
        time_val = data['evolution_time']
        success = data['ancilla_success_rate'] * 100
        
        print(f"{N:<6} {bond:<6} {method:<12} {energy:<10.2f} {time_val:<10.3f} {success:<10.1f}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    quantum_count = sum(1 for d in results.values() if d and d['method_used'] == 'quantum')
    fallback_count = sum(1 for d in results.values() if d and 'classical' in d['method_used'])
    
    print(f"\nQuantum successes: {quantum_count}/{len(results)}")
    print(f"Classical fallbacks: {fallback_count}/{len(results)}")
    
    if quantum_count > 0:
        avg_success = np.mean([d['ancilla_success_rate'] * 100 
                              for d in results.values() 
                              if d and d['method_used'] == 'quantum'])
        print(f"Average ancilla success rate: {avg_success:.2f}%")
    
    # Recommendations
    print("\n" + "="*60)
    print("L4 RECOMMENDATIONS")
    print("="*60)
    
    max_tested = max(results.keys())
    
    if quantum_count == 0:
        print("""
⚠️  All tests used classical fallback.

This means the quantum kernel is not achieving successful ancilla
post-selections. This is OK for Phase 2! You can:

1. Focus on classical GPU acceleration (CuPy)
2. Document the quantum debugging process
3. Show excellent classical results

OR investigate further:
- Try changing ancilla bit position (MSB vs LSB)
- Try different post-selection criteria ('0' vs '1')
- Adjust circuit parameters (trotter_steps, drift)
""")
    elif quantum_count < len(results):
        print(f"""
⚠️  Mixed results ({quantum_count} quantum, {fallback_count} fallback)

Quantum kernel works for some N values but not others.
Consider:
- Testing at smaller N first
- Increasing bond dimension for larger N
- Checking memory usage during failures
""")
    else:
        print(f"""
✓ All tests used quantum kernel successfully!

Your L4 setup is working well. Next steps:
- Scale up to N={max_tested + 4} or N={max_tested + 6}
- Compare energy quality vs classical
- Measure time-to-solution scaling
- Consider upgrading to A100 for N>30
""")

def generate_plots(results, prefix):
    """Generate plots from results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not found. Skipping plots.")
        return

    # Extract data
    ns = sorted([n for n in results if results[n]])
    if not ns: return

    q_times = [results[n]['evolution_time'] for n in ns]
    c_times = [results[n]['classical_time'] for n in ns]
    ratios = [results[n]['energy_ratio'] for n in ns]
    diffs = [results[n]['final_energy'] - results[n]['classical_energy'] for n in ns]
    
    # 1. Time Comparison (Verification)
    plt.figure(figsize=(10, 6))
    plt.plot(ns, q_times, 'o-', label='Quantum Time', linewidth=2, markersize=8)
    plt.plot(ns, c_times, 's--', label='Classical Budget', linewidth=2, markersize=8)
    plt.xlabel('System Size (N)', fontsize=12)
    plt.ylabel('Time (s)', fontsize=12)
    plt.title('Benchmark Time Budget', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{prefix}_time_vs_n.png', dpi=300)
    plt.close()
    print(f"Generated {prefix}_time_vs_n.png")
    
    # 2. Energy Ratio
    plt.figure(figsize=(10, 6))
    plt.plot(ns, ratios, 'd-', color='green', linewidth=2, markersize=8)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Parity (1.0)')
    plt.xlabel('System Size (N)', fontsize=12)
    plt.ylabel('Energy Ratio (Q / C)', fontsize=12)
    plt.title('Approximation Quality (Lower is Better)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{prefix}_ratio_vs_n.png', dpi=300)
    plt.close()
    print(f"Generated {prefix}_ratio_vs_n.png")

    # 3. Energy Difference
    plt.figure(figsize=(10, 6))
    plt.bar([str(n) for n in ns], diffs, color='purple', alpha=0.7)
    plt.xlabel('System Size (N)', fontsize=12)
    plt.ylabel('Energy Difference (Quantum - Classical)', fontsize=12)
    plt.title('Quantum Advantage Gap (Negative = Quantum Wins)', fontsize=14)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{prefix}_diff_vs_n.png', dpi=300)
    plt.close()
    print(f"Generated {prefix}_diff_vs_n.png")

def main():
    """Run L4-optimized tests."""
    import argparse
    parser = argparse.ArgumentParser(description='L4 GPU Test Script')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer shots)')
    parser.add_argument('--small', action='store_true', help='Test only small N (8,12)')
    parser.add_argument('--full', action='store_true', help='Test up to N=30 (may be slow)')
    args = parser.parse_args()
    
    print("CUDA-Q L4 GPU TEST")
    print("="*60)
    print("This script validates the quantum kernel on Brev L4 instances")
    print("="*60)
    
    # Environment check
    available_backends = l4_environment_check()
    
    if not HAS_CUDAQ:
        print("\n❌ CUDA-Q not available")
        print("This script requires CUDA-Q installation on L4 hardware")
        return
    
    if 'tensornet' not in available_backends:
        print("\n⚠️  Tensornet backend not available")
        print("Available backends:", available_backends)
    
    print("\n" + "="*60)
    
    # Determine N values to test
    if args.small:
        n_values = [8, 12]
    elif args.full:
        n_values = [8, 12, 16, 20, 24]
    else:
        n_values = [8, 12, 16, 20, 24]
    
    # Run tests
    try:
        print("\n🚀 Starting L4 quantum kernel tests...")
        results = test_quantum_kernel_l4(n_values, quick_mode=args.quick)
        
        if results:
            print_l4_summary(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'l4_results_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'gpu': 'L4',
                    'quick_mode': args.quick,
                    'results': results
                }, f, indent=2)
            
            print(f"\n📄 Results saved to {filename}")
            
            # Generate plots
            plot_prefix = f'benchmark_{timestamp}'
            generate_plots(results, plot_prefix)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()