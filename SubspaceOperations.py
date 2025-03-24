def IntersectionOfSpaces(B_1, B_2):
  # B_1 and B_2 are basis for two subspaces (of R^n): W_1 and W_2 
  # They are given as 2D numpy arrays (columns of the arrays form the basis)
  # This function returns a basis for the intersection of W_1 and W_2

def SumOfSpaces(basis_list):
  # the input basis_list consists of lists of numpy arrays (3D numpy array). basis_list[i] is a basis for a subspace W_i of R^n
  # This function returns a basis for the sum of W_1 + W_2 + ... + W_k

def OrthComplement(A,B)
  # A and B are basis for (2D numpy arrays) two subspaces W_A and W_B
  # This function assumes that W_B is a subspace of W_A
  # This function returns a basis for W_A intersecton (W_B)^perp
