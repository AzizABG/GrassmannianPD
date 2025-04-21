# test_computeGPD_example48.py

import numpy as np
import pytest
from compute_GPD import computeGPD

def normalize(v):
    return v / np.linalg.norm(v)

# Build the 4×4 distance matrix matching Example 4.8
D = np.array([
    [0.0, 0.0, 2.0, 3.0],   # 0–1=a@0, 0–2=tri@2, 0–3=d@3
    [0.0, 0.0, 1.0, 4.0],   # 1–2=b@1, 1–3=tri@4
    [2.0, 1.0, 0.0, 2.0],   # 2–3=c@2
    [3.0, 4.0, 2.0, 0.0]
], dtype=float)

k = 1  # tracking 1‑cycles

@pytest.mark.parametrize("l, r, lm, rm, expected", [
    # (0,∞)   → span[a]
    (0.0, 10.0, -1.0, 9.0, np.array([1.0, 0.0, 0.0, 0.0])),

    # (1,1)   → span[b – a]
    (1.0, 1.0, 0.0, 0.0, np.array([-1.0, 1.0, 0.0, 0.0])),

    # (1,2)   → span[2c – (a+b)]
    (1.0, 2.0, 0.0, 1.0, np.array([-1.0, -1.0, 2.0, 0.0])),

    # (3,4)   → span[3d – (a+b+c)]
    (3.0, 4.0, 2.0, 3.0, np.array([-1.0, -1.0, -1.0, 3.0])),
])
def test_example48_intervals(l, r, lm, rm, expected):
    Z = computeGPD(D, k, l, r, lm, rm)
    # Expect a single basis vector in R⁴
    assert Z.shape == (4, 1), f"expected one 4‑vector for interval ({l},{r})"
    v = normalize(Z[:, 0])
    e = normalize(expected)
    # allow overall sign‑flip
    assert np.allclose(np.abs(v), np.abs(e), atol=1e-6)


if __name__ == "__main__":
    # Run all tests in this file
    import sys
    sys.exit(pytest.main([__file__]))
