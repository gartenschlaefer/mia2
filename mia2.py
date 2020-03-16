# --
# library for mia2 - Music Information retrivAl 2

import numpy as np

def calc_pca(x):
  """
  calculate pca of signal, already ordered, n x m (samples x features)
  """

  # eigen stuff -> already sorted
  eig_val, eig_vec = np.linalg.eig(np.cov(x, rowvar=False))

  # pca transformation
  return np.dot(x, eig_vec)


