# --
# library for mia2 - Music Information retrivAl 2

import numpy as np


def calc_sdm(feat_frames, distance_measure='euclidean', emb=None):
  """
  calculate the self-distance matrix from frame feature vectors [n x m] i.e. [features x frames]
  """

  # init
  n, m = feat_frames.shape
  sdm = np.zeros((m, m))

  # run through each feature
  for i, feat in enumerate(feat_frames.T):

    # compare with each other feature frame
    for j in range(m):
      
      # check embedding
      if emb is None:

        # calculate distance
        sdm[i, j] = np.linalg.norm(feat - feat_frames[:, j])

      # do the embedding stuff
      else:

        # extend feat_frames with zeros
        e = np.hstack((feat_frames.copy(), np.zeros((n, emb+1))))

        # embedding implementation
        sdm[i, j] = np.linalg.norm(e[:, i:i+emb+1] - e[:, j:j+emb+1])

  return sdm


def calc_pca(x):
  """
  calculate pca of signal, already ordered, n x m (samples x features)
  """

  # eigen stuff -> already sorted
  eig_val, eig_vec = np.linalg.eig(np.cov(x, rowvar=False))

  # pca transformation
  return np.dot(x, eig_vec)


