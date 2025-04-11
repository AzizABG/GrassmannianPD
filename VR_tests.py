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