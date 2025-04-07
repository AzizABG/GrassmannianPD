def vietoris_rips(D, epsilon, k_max):
  #This function takes a distance matrix = D, epsilon, and the maximum degree k_max and computes the simplices in the epsilon-Vietoris-Rips complex for each degree 0,1,2,...,k_max 
  #(it might be better to do this for a fixed k in order to get rid of redundant computations. That is, implement for a single k, and if we need all 0,1,2,...,k_max, we can simply write a loop)


def boundary_maxrix(simplices_k, simplices_k_minus_1):
  #This function constructs the real-valued (numpy) boundary matrix from (k)-simplices to (k-1)-simplices.
