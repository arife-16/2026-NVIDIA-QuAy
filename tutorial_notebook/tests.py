import unittest
import numpy as np
import sys
import os

# Ensure local modules are importable
sys.path.append(os.getcwd())

from labs_utils import get_canonical, calculate_energy
from quantum_driver import HatanoNelsonDriver

class TestQuantumDriver(unittest.TestCase):
    
    def test_canonical_symmetry(self):
        """Test that get_canonical handles symmetries correctly."""
        # fixed seed for reproducibility
        np.random.seed(42)
        s = np.random.choice([1, -1], size=20)
        s_canon = get_canonical(s)
        
        # Variants
        variants = [s, -s, s[::-1], -s[::-1]]
        for v in variants:
            self.assertTrue(np.array_equal(get_canonical(v), s_canon), 
                            "Canonical form should be invariant under symmetry operations")
            
    def test_energy_invariance(self):
        """Test that energy is invariant under symmetry operations."""
        np.random.seed(42)
        s = np.random.choice([1, -1], size=20)
        e_orig = calculate_energy(s)
        s_canon = get_canonical(s)
        e_canon = calculate_energy(s_canon)
        self.assertEqual(e_orig, e_canon, "Energy should be invariant")
        
    def test_driver_shockwave(self):
        """Test that apply_shockwave runs and returns valid sequence."""
        N = 10
        driver = HatanoNelsonDriver(N)
        s = np.ones(N)
        s[0] = -1
        # reduced tau to prevent drift explosion during tests
        new_s, ipr = driver.apply_shockwave(s, tau=0.1)
        
        self.assertEqual(len(new_s), N)
        self.assertTrue(np.all(np.abs(new_s) == 1), "Sequence must be binary")
        
    def test_symmetry_projection(self):
        """Test that the driver enforces symmetry on the continuous state."""
        N = 10
        driver = HatanoNelsonDriver(N)
        # Create a non-symmetric state
        s = np.random.randn(N)
        s_proj = driver._project_to_symmetry(s)
        
        # Check if symmetric or skew-symmetric
        s_rev = s_proj[::-1]
        is_sym = np.allclose(s_proj, s_rev)
        is_skew = np.allclose(s_proj, -s_rev)
        
        self.assertTrue(is_sym or is_skew, "State must be projected to Symmetric or Skew-Symmetric subspace")

if __name__ == '__main__':
    unittest.main()
