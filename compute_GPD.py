from VietorisRips import boundary_matrix, vietoris_rips
from SubspaceOperations import SumOfSpaces, IntersectionOfSpaces, OrthComplement
import numpy as np
import scipy as sp
from scipy.linalg import null_space, qr

def computeGPD(D, k, l, r, lminus, rminus):
    ldict = vietoris_rips(D, l, k)
    lminusdict = vietoris_rips(D, lminus, k)
    rdict = vietoris_rips(D, r, k)
    rminusdict = vietoris_rips(D, rminus, k)

    

    #ZB(l, r)

    cycles = boundary_matrix(ldict[k], ldict[k-1])
    cycles_padded = pad_matrix_to_match_tuples(cycles, ldict[k-1], rdict[k-1])
    cycles_padded = pad_matrix_to_match_tuples(cycles_padded.T, ldict[k], rdict[k]).T
    cycles = null_space(cycles_padded)

    boundries = boundary_matrix(rdict[k+1], rdict[k])
    boundries, R = qr(boundries)

    ZBlr = IntersectionOfSpaces(cycles, boundries)

    #ZB(lminus, r)

    cycles = boundary_matrix(lminusdict[k], lminusdict[k-1])
    cycles_padded = pad_matrix_to_match_tuples(cycles, lminusdict[k-1], rdict[k-1])
    cycles_padded = pad_matrix_to_match_tuples(cycles_padded.T, lminusdict[k], rdict[k]).T
    cycles = null_space(cycles)

    boundries = boundary_matrix(rdict[k+1], rdict[k])
    boundries, R = qr(boundries)

    ZBlminusr = IntersectionOfSpaces(cycles, boundries)

    #ZB(l, rminus)

    cycles = boundary_matrix(ldict[k], ldict[k-1])
    cycles_padded = pad_matrix_to_match_tuples(cycles, ldict[k-1], rdict[k-1])
    cycles_padded = pad_matrix_to_match_tuples(cycles_padded.T, ldict[k], rdict[k]).T
    cycles = null_space(cycles)

    boundries = boundary_matrix(rminusdict[k+1], rminusdict[k])
    cycles_padded = pad_matrix_to_match_tuples(boundries, rminusdict[k], rdict[k])
    cycles_padded = pad_matrix_to_match_tuples(cycles_padded.T, rminusdict[k+1], rdict[k+1]).T
    boundries, R = qr(boundries)

    ZBlrminus = IntersectionOfSpaces(cycles, boundries)


    ZBalmostfinal = SumOfSpaces(ZBlminusr, ZBlrminus)

    ZBfinal = OrthComplement(ZBlr, ZBalmostfinal)


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
    

    return ZBfinal


def computeGPD_tester():
    D = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3)/2]
    ])

    k = 1  # We're interested in 1-dimensional holes (loops)
    l = 1.5
    r = 1.5
    lminus = 0.5
    rminus = 0.5

    ZBfinal = computeGPD(D, k, l, r, lminus, rminus)
    print("Test Case 1 Output Shape:", ZBfinal.shape)

    computeGPD_tester()
