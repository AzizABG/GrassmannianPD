from SubspaceOperations import IntersectionOfSpaces, THRESHOLD

import numpy as np
from scipy.linalg import null_space
import sympy as sp


def Intersection_tests():
    # Test 1: Complementary subspaces in R^2 (trivial intersection)
    B1 = np.array([[1.0], [0.0]])
    B2 = np.array([[0.0], [1.0]])
    inter = IntersectionOfSpaces(B1, B2)
    assert inter.shape[1] == 0, "Test 1 Failed: Expected no non-trivial intersection."
    print("Test 1 passed: Complementary subspaces have trivial intersection.")

    # Test 2: Identical subspaces in R^2 (full intersection)
    B1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    B2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    inter = IntersectionOfSpaces(B1, B2)
    # The intersection of identical subspaces should span the full space (dimension 2)
    assert inter.shape[1] == 2, "Test 2 Failed: Expected intersection dimension 2."
    rank_inter = np.linalg.matrix_rank(inter)
    assert rank_inter == 2, f"Test 2 Failed: Intersection vectors rank {rank_inter} != 2."
    print("Test 2 passed: Identical subspaces yield full intersection.")

    # Test 3: Non-trivial intersection in R^3 (intersection is a line)
    # Let B1 be spanned by {(1,0,0), (0,1,0)} and B2 by {(1,0,0), (0,0,1)}.
    # Their intersection should be the x-axis.
    B1 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 0.0]])
    B2 = np.array([[1.0, 0.0],
                   [0.0, 0.0],
                   [0.0, 1.0]])
    inter = IntersectionOfSpaces(B1, B2)
    assert inter.shape[1] == 1, "Test 3 Failed: Expected one-dimensional intersection."
    v = inter[:, 0]
    # Normalize the vector for a fair comparison (allowing a sign flip)
    norm_v = np.linalg.norm(v)
    assert norm_v > 0, "Test 3 Failed: Intersection vector has zero norm."
    v_norm = v / norm_v
    # Check if the normalized vector is (approximately) [1, 0, 0] or its negative
    if not (np.allclose(v_norm, np.array([1, 0, 0]), atol=1e-8) or 
            np.allclose(v_norm, np.array([-1, 0, 0]), atol=1e-8)):
        raise AssertionError("Test 3 Failed: Intersection vector is not aligned with the x-axis.")
    print("Test 3 passed: Non-trivial one-dimensional intersection in R^3 is correct.")

    # Test 4: B1 is an empty basis (zero columns)
    B1 = np.empty((3, 0))  # No basis vectors in R^3
    B2 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 0.0]])
    inter = IntersectionOfSpaces(B1, B2)
    assert inter.shape[1] == 0, "Test 4 Failed: Intersection with an empty B1 should be empty."
    print("Test 4 passed: Function handles empty B1 correctly.")

    # Test 5: B2 is an empty basis
    B1 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [0.0, 0.0]])
    B2 = np.empty((3, 0))  # Empty basis for B2
    inter = IntersectionOfSpaces(B1, B2)
    # Here, M = [B1] and if B1 has full column rank, its null space is empty.
    assert inter.shape[1] == 0, "Test 5 Failed: Intersection with an empty B2 should be empty if B1 is full rank."
    print("Test 5 passed: Function handles empty B2 correctly.")

    # Test 6: Nearly identical subspaces with small perturbations
    # Create two nearly identical bases in R^2 to ensure thresholding works
    B1 = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
    B2 = np.array([[1.0 + 1e-12, 0.0],
                   [0.0, 1.0 - 1e-12]])
    inter = IntersectionOfSpaces(B1, B2)
    # Expect a full two-dimensional intersection despite the tiny differences
    assert inter.shape[1] == 2, "Test 6 Failed: Nearly identical bases should yield full intersection."
    print("Test 6 passed: Near-threshold perturbations are handled correctly.")

    # Additional Robustness Test:
    # Verify that every intersection vector v lies in the span of B2.
    # For each computed intersection vector, solve B2 * y ≈ v.
    for test_num, (B1, B2) in enumerate([
            (np.array([[1.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 0.0]]),
             np.array([[1.0, 0.0],
                       [0.0, 0.0],
                       [0.0, 1.0]])),
            (np.array([[1.0, 0.0], [0.0, 1.0]]),
             np.array([[1.0, 0.0], [0.0, 1.0]]))
        ], start=7):
        inter = IntersectionOfSpaces(B1, B2)
        for i in range(inter.shape[1]):
            v = inter[:, i]
            # Solve least squares for y: B2 y = v.
            y, residuals, rank, s = np.linalg.lstsq(B2, v, rcond=None)
            recon = B2 @ y
            assert np.linalg.norm(recon - v) < 1e-8, f"Test {test_num} Failed: Intersection vector {i} is not in span(B2)."
        print(f"Test {test_num} passed: Verified intersection vectors lie in span(B2).")

    print("Test 9: 3D subspaces in ℝ⁶ with a 1D intersection ===")
    B2 = np.column_stack([[3,6,3,0,3,-3], [1,0,-1,0,0,0], [0,1,0,1,0,0]])
    B1 = np.column_stack([[1, 3, 1, 1, 1, 0], [1, 1, 1, -1, 1, -2], [1, 0, -1, 2, -1, 1]])
    # The expected common intersection is span(v).
    expected_v = [1,2,1,0,1,-1]
    
    inter = IntersectionOfSpaces(B1, B2)
    
    print("Intersection dimension (expected 1):", inter.shape[1])
    
    # Verify collinearity with expected_v.
    if inter.shape[1] >= 1:
        for i in range(inter.shape[1]):
            vec = inter[:, i]
            # Compare the ratios of corresponding entries (ignoring zeros in expected_v).
            ratios = []
            for a, b in zip(vec, expected_v):
                if abs(b) > THRESHOLD:
                    ratios.append(a / b)
            if len(ratios) > 0 and np.allclose(ratios, ratios[0], atol=1e-8):
                print(f"Intersection vector {i} is collinear with v (scale factor ~ {ratios[0]:.4f}).")
            else:
                print(f"Intersection vector {i} is NOT collinear with v.")



    

    print("test 10: 3D subspaces in ℝ⁶ with a 2D intersection ===")
    # Here we want the two subspaces to share a common 2D plane.
    # Define two independent vectors that span the common plane.
    v1 = np.array([1, 2, 0, 1, 0, 3], dtype=float)
    v2 = np.array([0, 1, 1, 0, 2, 1], dtype=float)
    # The expected intersection is span{v1, v2}.
    
    B3 = np.column_stack([[1,4,2,1,4,5], [3,5,-1,3,-2,8], [1, 0, 1, 0, 1, 0]])
    
    B4 = np.column_stack([[2,3,-1,2,-2,5], [1,3,1,1,2,4], [0, 1, 0, 1, 0, 1]])
    
    inter = IntersectionOfSpaces(B3, B4)

    print("Intersection dimension (expected 2):", inter.shape[1])
    
    # Verify that each computed intersection vector lies in span{v1, v2}.
    CommonPlane = np.column_stack([v1, v2])
    if inter.shape[1] >= 1:
        for i in range(inter.shape[1]):
            vec = inter[:, i]
            # Solve for coefficients a, b:  a*v1 + b*v2 ≈ vec.
            coeffs, residuals, rank, s = np.linalg.lstsq(CommonPlane, vec, rcond=None)
            recon = CommonPlane @ coeffs
            if np.allclose(recon, vec, atol=1e-8):
                print(f"Intersection vector {i} lies in the expected 2D plane (coeffs: {coeffs}).")
            else:
                print(f"Intersection vector {i} does NOT lie in the expected 2D plane.")



Intersection_tests()