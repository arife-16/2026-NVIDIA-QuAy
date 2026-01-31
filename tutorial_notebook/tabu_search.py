import numpy as np
import warnings
try:
    from labs_utils import calculate_energy, calculate_autocorrelations
except ImportError:
    try:
        from labs_utils import calculate_energy, calculate_autocorrelations
    except ImportError:
        # Inline fallback for Tabu
        def calculate_autocorrelations(sequence):
            N = len(sequence)
            autocorrelations = []
            for k in range(1, N):
                C_k = np.dot(sequence[:-k], sequence[k:])
                autocorrelations.append(C_k)
            return np.array(autocorrelations)
        def calculate_energy(sequence):
            C_k = calculate_autocorrelations(sequence)
            return np.sum(C_k**2)

# Try to import cupy, else use numpy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False
    warnings.warn("CuPy not found in TabuSearch. Using NumPy.")

class TabuSearch:
    def __init__(self, N, min_tenure=3, max_tenure=20, max_iter=1000):
        self.N = N
        self.min_tenure = min_tenure
        self.max_tenure = max_tenure
        self.max_iter = max_iter
        self.tabu_list = {} # {index: expiry_iter}
        
    def _calculate_batch_delta_energy(self, sequence, correlations):
        """
        Vectorized Delta Energy Calculation.
        Calculates dE for all N possible bit flips in parallel.
        Returns array of delta_E of shape (N,)
        """
        # Convert to appropriate backend
        if HAS_CUPY:
            s = cp.asarray(sequence)
            C = cp.asarray(correlations)
        else:
            s = np.asarray(sequence)
            C = np.asarray(correlations)
            
        N = self.N
        
        # We need to compute delta_E for each flip k in [0, N-1]
        # The formula for one flip k is:
        # change_k = -2 * s[k]
        # delta_E_k = sum_{tau=1}^{N-1} [ (C_tau + delta_C_tau_k)^2 - C_tau^2 ]
        #           = sum_{tau} [ 2 * C_tau * delta_C_tau_k + (delta_C_tau_k)^2 ]
        
        # delta_C_tau_k = change_k * (s[k-tau] + s[k+tau]) (with boundary checks)
        
        # Let's vectorize this.
        # We can construct a matrix M of shape (N, N-1) where M[k, tau-1] = delta_C_tau_k / change_k
        # M[k, tau-1] = s[k-tau] (if valid) + s[k+tau] (if valid)
        
        # Construct M efficiently
        # M corresponds to the "interaction" of bit k with lag tau.
        # This matrix M depends only on sequence s.
        
        # Create shifted versions of s
        # We need s[k-tau] and s[k+tau]
        # This is essentially a convolution or just indexing.
        
        # We can build M using broadcasting or sliding windows?
        # N is small (40-200), so N^2 is tiny. We can build full matrices.
        
        # Create indices
        k_indices = cp.arange(N).reshape(N, 1) if HAS_CUPY else np.arange(N).reshape(N, 1)
        tau_indices = cp.arange(1, N).reshape(1, N-1) if HAS_CUPY else np.arange(1, N).reshape(1, N-1)
        
        # Compute indices for s[k-tau] and s[k+tau]
        idx_minus = k_indices - tau_indices # Shape (N, N-1)
        idx_plus = k_indices + tau_indices  # Shape (N, N-1)
        
        # 3Create Mask for valid indices
        mask_minus = (idx_minus >= 0)
        mask_plus = (idx_plus < N)
        
        # Gather values
        # We need to handle out of bounds. 
        
        val_minus = cp.zeros((N, N-1), dtype=s.dtype) if HAS_CUPY else np.zeros((N, N-1), dtype=s.dtype)

        # Safe indexing:
        safe_idx_minus = cp.where(mask_minus, idx_minus, 0) if HAS_CUPY else np.where(mask_minus, idx_minus, 0)
        term_minus = s[safe_idx_minus]
        term_minus = cp.where(mask_minus, term_minus, 0) if HAS_CUPY else np.where(mask_minus, term_minus, 0)
        
        safe_idx_plus = cp.where(mask_plus, idx_plus, 0) if HAS_CUPY else np.where(mask_plus, idx_plus, 0)
        term_plus = s[safe_idx_plus]
        term_plus = cp.where(mask_plus, term_plus, 0) if HAS_CUPY else np.where(mask_plus, term_plus, 0)
        
        M = term_minus + term_plus # Shape (N, N-1)
        
        # Now compute delta_C for each k, tau
        # change_k vector: -2 * s[k]
        change = -2 * s # Shape (N,)
        change_col = change.reshape(N, 1)
        
        delta_C = M * change_col # Shape (N, N-1)
        
        # compute energy change
        # sum_{tau} [ 2 * C_tau * delta_C + delta_C^2 ]
        
        C_row = C.reshape(1, N-1) # C corresponds to tau=1..N-1
        
        # Term 1: 2 * C * delta_C
        term1 = 2 * C_row * delta_C
        
        # Term 2: delta_C^2
        term2 = delta_C ** 2
        
        # Sum over tau (axis 1)
        dE_values = cp.sum(term1 + term2, axis=1) if HAS_CUPY else np.sum(term1 + term2, axis=1)
        
        return dE_values

    def solve(self, initial_sequence=None, callback=None):
        if initial_sequence is None:
            current_seq = np.random.choice([1, -1], size=self.N)
        else:
            current_seq = np.array(initial_sequence).copy()
            
        best_seq = current_seq.copy()
        
        # Initial Full Calculation
        current_correlations = calculate_autocorrelations(current_seq)
        current_energy = np.sum(current_correlations**2)
        
        best_energy = current_energy
        
        self.tabu_list = {} 
        stagnation_counter = 0
        
        for it in range(self.max_iter):
            # If we are stagnating, increase "memory" to force the solver away.
            # If we are improving, relax memory to allow fine-tuning.
            # Linear scaling: min + (max-min) * (stag / threshold)
            
            stag_ratio = min(1.0, stagnation_counter / 50.0)
            current_tenure = int(self.min_tenure + (self.max_tenure - self.min_tenure) * stag_ratio)
            
            # Vectorized Evaluation
            dE_values = self._calculate_batch_delta_energy(current_seq, current_correlations)
            
            # Convert back to CPU if needed for logic (small array N)
            if HAS_CUPY:
                dE_values_cpu = cp.asnumpy(dE_values)
            else:
                dE_values_cpu = dE_values
            
            best_move_delta = float('inf')
            best_move_idx = -1
            
            
            # Create a mask of tabu indices
            is_tabu = np.zeros(self.N, dtype=bool)
            for idx, expiry in self.tabu_list.items():
                if expiry > it:
                    is_tabu[idx] = True
            
            # Aspiration: dE + current_energy < best_energy
            potential_energies = current_energy + dE_values_cpu
            is_better_than_best = potential_energies < best_energy
            
            # Allow move if (Not Tabu) OR (Aspiration Met)
            allowed_moves = (~is_tabu) | is_better_than_best
            
            if np.any(allowed_moves):
                # Find best allowed move
                # Mask disallowed moves with infinity
                # Ensure float type
                masked_dE = dE_values_cpu.astype(float)
                masked_dE[~allowed_moves] = float('inf')
                
                best_move_idx = np.argmin(masked_dE)
                best_move_delta = masked_dE[best_move_idx]
                
                if best_move_delta == float('inf'):
                     best_move_idx = -1 # Should not happen if any allowed
            else:
                best_move_idx = -1
            
            # Execute Move
            if best_move_idx != -1:
                # Update Sequence
                current_seq[best_move_idx] *= -1
                
                # Update Energy
                current_energy += best_move_delta
                
                # Recompute correlations
                # Actually, delta update of correlations is faster O(N).
                # But we have O(N^2) vectorized calc anyway, so full recompute O(N^2) is fine.
                # Same cost.
                current_correlations = calculate_autocorrelations(current_seq)
                
                self.tabu_list[best_move_idx] = it + current_tenure
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_seq = current_seq.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1
            
            if callback:
                stop = callback(current_seq, current_energy, it, stagnation_counter)
                if stop:
                    break
                    
        return best_seq, best_energy
