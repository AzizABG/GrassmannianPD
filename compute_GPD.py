from VietorisRips import boundary_matrix, vietoris_rips
from SubspaceOperations import SumOfSpaces, IntersectionOfSpaces, OrthComplement
import numpy as np
import scipy as sp
from scipy.linalg import null_space, qr

def computeGPD(D, k, l, r):

    upper_dists = D[np.triu_indices_from(D, k=1)]
    
    # Get the sorted unique distances
    unique_dists = np.unique(upper_dists)
    
    # Find l-1 and r-1: the largest values strictly less than l and r
    lminus = max(unique_dists[unique_dists < l], default=0)
    rminus = max(unique_dists[unique_dists < r], default=0)

    def pad_matrix_to_match_tuples(mat: np.ndarray, short_list: list, full_list: list) -> np.ndarray:

        assert len(mat) == len(short_list), "Matrix row count must match short_list length"
        padded_rows = []
        i = 0  # index for short_list / mat
        j = 0  # index for full_list

        while j < len(full_list):
            if i < len(short_list) and short_list[i] == full_list[j]:
                padded_rows.append(mat[i])
                i += 1
            else:
                # Add a row of zeros with the same number of columns as `mat`
                padded_rows.append(np.zeros(mat.shape[1]))
            j += 1

        return np.array(padded_rows)
    
    print(lminus)
    ldict = vietoris_rips(D, l, k+1)
    lminusdict = vietoris_rips(D, lminus, k+1)
    rdict = vietoris_rips(D, r, k+1)
    rminusdict = vietoris_rips(D, rminus, k+1)

    if k+1 not in rdict:
        return 0

    #ZB(l, r)
    if k not in ldict or k-1 not in ldict:
        ZBlr = np.zeros((len(rdict[k]), 1))
    else:
        cycles = boundary_matrix(ldict[k], ldict[k-1])
        cycles = null_space(cycles)
        cycles_padded = pad_matrix_to_match_tuples(cycles, ldict[k], rdict[k])
    
        boundries = boundary_matrix(rdict[k+1], rdict[k])
        boundries, R = qr(boundries)

        ZBlr = IntersectionOfSpaces(cycles_padded, boundries)
        print("look here for ZBlr")
        print(ZBlr)

    #ZB(lminus, r)
    if k not in lminusdict or k-1 not in lminusdict:
        ZBlminusr = np.zeros((len(rdict[k]), 1))
    else:
        cycles = boundary_matrix(lminusdict[k], lminusdict[k-1])
        cycles = null_space(cycles)
        cycles_padded = pad_matrix_to_match_tuples(cycles, lminusdict[k], rdict[k])
        

        boundries = boundary_matrix(rdict[k+1], rdict[k])
        boundries, R = qr(boundries)

        ZBlminusr = IntersectionOfSpaces(cycles_padded, boundries)

    #ZB(l, rminus)
    if k not in ldict or k-1 not in ldict or k+1 not in rminusdict:
        ZBlrminus = np.zeros((len(rdict[k]), 1))
    else:
        cycles = boundary_matrix(ldict[k], ldict[k-1])
        cycles = null_space(cycles)
        cycles_padded = pad_matrix_to_match_tuples(cycles, ldict[k], rdict[k])
        

        boundries = boundary_matrix(rminusdict[k+1], rminusdict[k])
        boundries, R = qr(boundries)
        boundries_padded = pad_matrix_to_match_tuples(boundries, rminusdict[k], rdict[k])
        

        ZBlrminus = IntersectionOfSpaces(cycles_padded, boundries_padded)


    ZBalmostfinal = SumOfSpaces(np.stack([ZBlminusr, ZBlrminus], axis=0))

    print(ZBlr)
    print("here is ZBALMOSTNFINAL")
    print(ZBalmostfinal)

    if ZBalmostfinal.size == 0:
        print("HELOOOOOO")
        return ZBlr
    ZBfinal = OrthComplement(ZBlr, ZBalmostfinal)
    

    return ZBfinal


def computeGPD_tester():
    D = np.array([
    [0, 1, 1, 2],
    [1, 0, 2, 1],
    [1, 2, 0, 1],
    [2, 1, 1, 0]
    ])

    k = 1  # We're interested in 1-dimensional holes (loops)
    l = 1
    r = 2
    lminus = 0
    rminus = 1

    ZBfinal = computeGPD(D, k, l, r)
    print("Test Case 1 Output Shape:", ZBfinal.shape)

computeGPD_tester()

