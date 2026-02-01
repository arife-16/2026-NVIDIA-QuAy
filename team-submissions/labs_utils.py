#!/usr/bin/env python3
"""
Comprehensive Type-Safe Wrapper for LABS Utilities

This module provides type-safe wrappers around the LABS utility functions
to ensure compatibility with both NumPy and CuPy arrays, and proper
integration with CUDA-Q.
"""

import numpy as np
import warnings

# Force NumPy as requested by user
HAS_CUPY = False
# We import numpy as cp to satisfy code that might reference cp, 
# though usually we guard with HAS_CUPY.
import numpy as cp 

def is_cupy_array(arr):
    """
    Reliably detect if an array is a CuPy array.
    """
    return False


def to_numpy(arr):
    """
    Convert array to NumPy.
    """
    return np.asarray(arr)


def to_cupy(arr):
    """
    Convert array to CuPy if available, otherwise return NumPy.
    """
    return np.asarray(arr)


def calculate_autocorrelations(sequence):
    """
    Calculate the autocorrelation of a sequence for all lags k=1 to N-1.
    Uses FFT for O(N log N) complexity.
    """
    # Always use NumPy
    xp = np
    
    seq = np.asarray(sequence)
    
    # Zero-pad to length >= 2N - 1 to avoid circular convolution
    N = len(seq)
    size = 2 * N
    
    # Pad with zeros
    s_padded = xp.zeros(size, dtype=seq.dtype)
    s_padded[:N] = seq
    
    # FFT
    S = xp.fft.fft(s_padded)
    
    # Power Spectrum
    P = S * xp.conj(S)
    
    # IFFT
    R = xp.fft.ifft(P)
    
    # Extract real part
    R = xp.real(R)
    
    # Return lags 1 to N-1
    return R[1:N]


def get_canonical(sequence):
    """
    Returns the canonical form of a sequence under Flip and Reversal symmetries.
    Always returns a NumPy array for consistency.
    """
    # Convert to NumPy for canonical form computation
    s = to_numpy(sequence)
    s_neg = -s
    s_rev = s[::-1]
    s_rev_neg = -s[::-1]
    
    # Create list of all 4 variants
    variants = [s, s_neg, s_rev, s_rev_neg]
    
    # Convert to tuples for sorting
    var_tuples = [tuple(v) for v in variants]
    best_tuple = min(var_tuples)
    
    return np.array(best_tuple)


def calculate_merit_factor(sequence):
    """
    Calculate the Merit Factor F of a binary sequence.
    F = N^2 / (2 * sum(C_k^2))
    
    Returns a Python float for consistency.
    """
    N = len(sequence)
    C_k = calculate_autocorrelations(sequence)
    
    sidelobe_energy = np.sum(C_k**2)
    
    if sidelobe_energy == 0:
        return float('inf')
        
    F = N**2 / (2 * sidelobe_energy)
    return float(F)


def calculate_energy(sequence):
    """
    Calculate the energy (sum of squared sidelobes).
    This is the quantity to minimize.
    
    Returns a Python float for consistency.
    """
    C_k = calculate_autocorrelations(sequence)
    
    energy = np.sum(C_k**2)
    return float(energy)


def check_barker13():
    """
    Verify the calculation with Barker-13 sequence.
    """
    barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    
    mf = calculate_merit_factor(barker13)
    energy = calculate_energy(barker13)
    
    print(f"Barker-13 Sequence: {barker13}")
    print(f"Merit Factor: {mf}")
    print(f"Energy (Sidelobes): {energy}")
    
    return mf


if __name__ == "__main__":
    check_barker13()
