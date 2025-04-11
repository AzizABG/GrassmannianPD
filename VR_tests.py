import numpy as np
from scipy.linalg import null_space
import sympy as sp

from VietorisRips import vietoris_rips

def VR_test():
    x = np.array([[0, 2, 3], 
                 [2, 0, 4],
                 [3, 4, 0]])
    epsilon = 2
    k = 0

    print(vietoris_rips(x, epsilon, k))

VR_test()