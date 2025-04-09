#look into scipy linalg operations
#anything less than 10^-10 should be 0, should be able to do this with null_mask parameter
import numpy as np
from scipy.linalg import null_space
import sympy as sp

THRESHOLD = 10**(-10)

def IntersectionOfSpaces(B_1, B_2):
     # Form the matrix [B_1, -B_2]
    M = np.hstack([B_1, -B_2])
    
    # Compute the null space of M for coefficients [x; y]
    ns = null_space(M, rcond=THRESHOLD)
    ns[np.abs(ns) < THRESHOLD] = 0  # Zero-out near-zero values
    
    # The first part of each nullspace vector corresponds to coefficients for B_1
    k1 = B_1.shape[1]
    x_coeff = ns[:k1, :]
    
    # Filter out any vectors that are trivial in the x-part (i.e., yield zero in the intersection)
    valid_indices = [i for i in range(x_coeff.shape[1]) if np.linalg.norm(x_coeff[:, i]) > 0]
    if len(valid_indices) == 0:
        return np.zeros((B_1.shape[0], 0))
    
    # Compute the intersection vectors as B_1 * (x coefficients)
    intersection_vectors = B_1 @ x_coeff[:, valid_indices]
    intersection_vectors[np.abs(intersection_vectors) < THRESHOLD] = 0
    
    return intersection_vectors







  # B_1 and B_2 are basis for two subspaces (of R^n): W_1 and W_2 
  # They are given as 2D numpy arrays (columns of the arrays form the basis)
  # This function returns a basis for the intersection of W_1 and W_2
  #horizontally stack and then row reduce

def SumOfSpaces(basis_list):
  all_vectors = basis_list.reshape(-1, basis_list.shape[-1])
  M = sp.Matrix(all_vectors)
  M_rref = M.rref()[0]


  M_rref_np = np.array(M_rref.tolist(), dtype=float)
  M_rref_np[np.abs(M_rref_np) < THRESHOLD] = 0
  
  # Extract nonzero rows
  basis = [row for row in M_rref_np if np.any(np.abs(row) > THRESHOLD)]
  
  return np.array(basis)



  # the input basis_list consists of lists of numpy arrays (3D numpy array). basis_list[i] is a basis for a subspace W_i of R^n
  # This function returns a basis for the sum of W_1 + W_2 + ... + W_k
  #vertically stack and then row reduce

def OrthComplement(A, B):
    # Number of rows of A
    n = A.shape[0]
    
    # Orthonormalize B (if B is empty, we treat Q_B as empty so every vector is in B⊥)
    if B.shape[1] > 0:
        Q_B, _ = np.linalg.qr(B)
    else:
        Q_B = np.empty((n, 0))  # Empty B implies every vector is in B⊥

    # Orthonormalize the entire span of A (considering A as a set of vectors)
    if A.shape[1] > 0:
        Q_A, _ = np.linalg.qr(A)
    else:
        Q_A = np.empty((n, 0))  # Empty A implies we return the whole space

    # Combine Q_B and Q_A to form a combined orthonormal basis
    Q_combined = np.hstack((Q_B, Q_A))

    # Perform QR decomposition on the combined matrix to get an orthonormal basis of the combined span
    Q_combined, _ = np.linalg.qr(Q_combined)

    # Now the orthogonal complement to B and A will be the remaining part in the orthonormalized space
    Q_complement = Q_combined[:, Q_B.shape[1]:]

    return Q_complement


def test_OrthComplement():
    A = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [0, 0, 1]], dtype=float).T  # 3 vectors in ℝ³

    B = np.array([[1, 0, 0],
              [0, 1, 0], ], dtype=float).T  # shape: (3, 2)

    # Compute the orthogonal complement of span(A) relative to span(B)
    Q = OrthComplement(A, B)

    print("Orthonormal basis of vectors in A orthogonal to B:")
    print(Q)

    # Check orthogonality: Q.T @ B should be (approximately) zero
    print("Q.T @ B (should be close to zero):")
    print(Q.T @ B)

    # Check orthonormality of result
    print("Q.T @ Q (should be identity):")
    print(Q.T @ Q)

test_OrthComplement()
