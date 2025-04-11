import numpy as np
from scipy.linalg import null_space
import sympy as sp

from VietorisRips import vietoris_rips

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

    print(vietoris_rips(D, epsilon, k))

VR_test()