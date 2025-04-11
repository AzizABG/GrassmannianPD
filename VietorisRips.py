from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from itertools import combinations
from collections import defaultdict
import numpy as np
from scipy.linalg import null_space
import sympy as sp



def vietoris_rips(D, epsilon, k_max):
    """
    Computes the Vietoris-Rips complex up to dimension k_max
    for a given distance matrix D and epsilon.
    Returns a dictionary mapping each dimension to a list of simplices.
    """
    n = len(D)
    complex_dict = defaultdict(list)

    # 0-simplices: vertices
    for i in range(n):
        complex_dict[0].append((i,))

    # 1-simplices: edges with length ≤ ε
    adjacency = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i][j] <= epsilon:
                edge = (i, j)
                complex_dict[1].append(edge)
                adjacency[i].add(j)
                adjacency[j].add(i)

    # Higher-order simplices
    def is_clique(vertices):
        return all(j in adjacency[i] for i, j in combinations(vertices, 2))

    for k in range(2, k_max + 1):
        for combo in combinations(range(n), k + 1):
            if is_clique(combo):
                filtration_value = max(D[i][j] for i, j in combinations(combo, 2))
                if filtration_value <= epsilon:
                    complex_dict[k].append(combo)

    return dict(complex_dict)




def boundary_matrix(simplices_k, simplices_k_minus_1):
    """
    Constructs the real-valued boundary matrix from k-simplices to (k-1)-simplices.
    """
    # Mapping from simplex to index for (k-1)-simplices
    index_map = {tuple(sorted(s)): i for i, s in enumerate(simplices_k_minus_1)}
    n_rows = len(simplices_k_minus_1)
    n_cols = len(simplices_k)

    B = np.zeros((n_rows, n_cols))

    for j, simplex in enumerate(simplices_k):
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i+1:]  # remove one vertex
            sign = (-1) ** i  # orientation sign
            if face in index_map:
                row = index_map[face]
                B[row, j] = sign

    return B
#   #This function constructs the real-valued (numpy) boundary matrix from (k)-simplices to (k-1)-simplices.
