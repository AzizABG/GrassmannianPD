from scipy.spatial.distance import pdist, squareform
from itertools import combinations


def vietoris_rips(D, epsilon, k_max):
  #This function takes a distance matrix = D, epsilon, and the maximum degree k_max and computes the simplices in the epsilon-Vietoris-Rips complex for each degree 0,1,2,...,k_max 
  #(it might be better to do this for a fixed k in order to get rid of redundant computations. That is, implement for a single k, and if we need all 0,1,2,...,k_max, we can simply write a loop)
  n = len(D)
  print(n)

  # 0-simplices: vertices
  simplices = [tuple([i]) for i in range(n)]

  # 1-simplices: edges with length ≤ ε
  edges = []
  for i in range(n):
      for j in range(i + 1, n):
          d = D[i][j]
          print(d)
          if d <= epsilon:
              edges.append(tuple(sorted([i, j])))
  simplices += edges

  # Higher-order simplices
  from collections import defaultdict
  adjacency = defaultdict(set)
  for (i, j) in edges:
      adjacency[i].add(j)
      adjacency[j].add(i)

  def is_clique(vertices):
      return all(j in adjacency[i] for i, j in combinations(vertices, 2))

  for k in range(2, k_max + 1):
      for combo in combinations(range(n), k + 1):
          if is_clique(combo):
              filtration_value = max(D[i][j] for i, j in combinations(combo, 2))
              if filtration_value <= epsilon:
                  simplices.append((combo))

  return simplices  # List of (simplex, filtration value)




# def boundary_matrix(simplices_k, simplices_k_minus_1):
#   #This function constructs the real-valued (numpy) boundary matrix from (k)-simplices to (k-1)-simplices.
