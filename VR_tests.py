import numpy as np
from scipy.linalg import null_space
import sympy as sp

from VietorisRips import vietoris_rips, boundary_matrix


def VR_test():
    D = np.array([
    [0,   1,   1,   1.4, 3.0, 1.1],
    [1,   0,   1,   1.4, 2.8, 1.0],
    [1,   1,   0,   1.4, 2.7, 1.0],
    [1.4, 1.4, 1.4, 0,   2.9, 1.3],
    [3.0, 2.8, 2.7, 2.9, 0,   3.2],
    [1.1, 1.0, 1.0, 1.3, 3.2, 0  ]
])
    epsilon = 1.5
    k = 3

    rips_complex = vietoris_rips(D, epsilon, k)
    print(rips_complex)
    simplices_2 = rips_complex[2]
    simplices_1 = rips_complex[1]

# Call the boundary matrix function
    B2 = boundary_matrix(simplices_2, simplices_1)
    print(B2)
VR_test()


def VR_test2():
    D = np.array([
    [0,2,3,4],
    [2,0,2,4],
    [3,2,0,1],
    [4,4,1,0]
])
    epsilon = 1.5
    k = 3

    rips_complex = vietoris_rips(D, epsilon, k)
    print(rips_complex)
    simplices_2 = rips_complex[1]
    simplices_1 = rips_complex[0]

# Call the boundary matrix function
    B2 = boundary_matrix(simplices_2, simplices_1)
    print(B2)
VR_test2()

def run_test(test_id, simplices_k, simplices_k_minus_1, expected):
    result = boundary_matrix(simplices_k, simplices_k_minus_1)
    if np.array_equal(result, expected):
        print(f"Test {test_id} passed")
    else:
        print(f"Test {test_id} failed")
        print("Expected:")
        print(expected)
        print("Got:")
        print(result)


# Test 1: Single triangle
simplices_1 = [(0, 1), (0, 2), (1, 2)]
simplices_2 = [(0, 1, 2)]
expected_1 = np.array([[1], [-1], [1]])
run_test(1, simplices_2, simplices_1, expected_1)

# Test 2: Two triangles sharing an edge
simplices_1 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
simplices_2 = [(0, 1, 2), (1, 2, 3)]
expected_2 = np.array([
    [ 1.,  0.],
    [-1.,  0.],
    [ 1.,  1.],
    [ 0., -1.],
    [ 0.,  1.]
])
run_test(2, simplices_2, simplices_1, expected_2)

# Test 3: Edges to vertices
simplices_0 = [(0,), (1,), (2,)]
simplices_1 = [(0, 1), (0, 2), (1, 2)]
expected_3 = np.array([
    [-1., -1.,  0.],
    [ 1.,  0., -1.],
    [ 0.,  1.,  1.]
])
run_test(3, simplices_1, simplices_0, expected_3)

# Test 4: Empty input
simplices_0 = []
simplices_1 = []
expected_4 = np.zeros((0, 0))
run_test(4, simplices_1, simplices_0, expected_4)

# Test 5: One edge to one vertex (should fail: invalid simplex)
simplices_0 = [(0,), (1,)]
simplices_1 = [(0, 2)]  # 2 isn't in any vertex
expected_5 = np.array([
    [-1.],
    [ 0.]  # this will likely fail, but checks robustness
])
run_test(5, simplices_1, simplices_0, expected_5)
