import numpy as np

def calculate_autocorrelations(sequence):
    """
    Calculate the autocorrelation of a sequence for all lags k=1 to N-1.
    Uses FFT for O(N log N) complexity, optimized for GPU if available.
    Sequence should be a numpy array (or cupy array) of 1s and -1s.
    """
    try:
        import cupy as cp
        HAS_CUPY = True
    except ImportError:
        import numpy as cp
        HAS_CUPY = False
        
    is_cupy = HAS_CUPY and hasattr(sequence, 'device')
    
    xp = cp if is_cupy else np
    
    # Zero-pad to length >= 2N - 1 to avoid circular convolution
    N = len(sequence)
    size = 2 * N
    
    # Compute FFT
    # Note: If sequence is real, we can use rfft for efficiency, but standard fft is fine.
    # We want Aperiodic Autocorrelation.
    # R[k] = sum_i s[i] * s[i+k]
    # This corresponds to IFFT(|FFT(s)|^2) if properly padded.
    
    # Pad with zeros
    s_padded = xp.zeros(size)
    s_padded[:N] = sequence
    
    # FFT
    S = xp.fft.fft(s_padded)
    
    # Power Spectrum (Corresponds to Correlation)
    # Correlation is Convolution of s(t) and s(-t).
    # In freq domain: S(f) * conjugate(S(f)) = |S(f)|^2
    P = S * xp.conj(S)
    
    # IFFT
    R = xp.fft.ifft(P)
    
    # Extract real part (should be real for autocorrelation)
    R = xp.real(R)
    
    # The result R contains correlations for lags:
    # R[0] = lag 0 (Energy/Norm^2)
    # R[1] = lag 1
    # ...
    # R[N-1] = lag N-1
    # Note: Due to standard FFT ordering and zero padding, the positive lags are at the beginning.
    return R[1:N]

def get_canonical(sequence):
    """
    Returns the canonical form of a sequence under Flip and Reversal symmetries.
    Example: For S, returns min(S, -S, S_reversed, -S_reversed) lexicographically.
    """
    s = np.array(sequence)
    s_neg = -s
    s_rev = s[::-1]
    s_rev_neg = -s[::-1]
    
    # Create a list of all 4 variants
    variants = [s, s_neg, s_rev, s_rev_neg]
    
    var_tuples = [tuple(v) for v in variants]
    best_tuple = min(var_tuples)
    
    return np.array(best_tuple)

def calculate_merit_factor(sequence):
    """
    Calculate the Merit Factor F of a binary sequence.
    F = N^2 / (2 * sum(C_k^2))
    """
    N = len(sequence)
    C_k = calculate_autocorrelations(sequence)
    sidelobe_energy = np.sum(C_k**2)
    
    if sidelobe_energy == 0:
        return float('inf') # Perfect sequence (Barker-N for N<=13 sometimes)
        
    F = N**2 / (2 * sidelobe_energy)
    return F

def calculate_energy(sequence):
    """
    Calculate the energy (sum of squared sidelobes).
    This is the quantity to minimize.
    """
    C_k = calculate_autocorrelations(sequence)
    return np.sum(C_k**2)

def check_barker13():
    """
    Verify the calculation with Barker-13 sequence.
    Barker 13: + + + + + - - + + - + - +
    """
    # Barker 13 pattern: +1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1
    barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    
    mf = calculate_merit_factor(barker13)
    energy = calculate_energy(barker13)
    
    print(f"Barker-13 Sequence: {barker13}")
    print(f"Merit Factor: {mf}")
    print(f"Energy (Sidelobes): {energy}")
    
    # Barker 13 is known to have Merit Factor 13^2 / (2*6) approx 14.08
    return mf

if __name__ == "__main__":
    check_barker13()
