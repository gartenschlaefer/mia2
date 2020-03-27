# --
# library for mia2 - Music Information retrivAl 2

import numpy as np


def calc_recurrence_matrix(sdm, w=4):
  """
  calc the recurrence matrix from the sdm matrix with the template matching algorithm - ncc
  """
  
  from skimage.util import view_as_windows

  # init
  R = np.zeros(sdm.shape) 

  # shape
  m, n = sdm.shape

  # get templates from sdm matrix
  Templates = np.squeeze(view_as_windows(np.pad(sdm, ((0, 0), (0, w-1))), (m, w), step=1)).reshape(m, n*w)

  # substract mean
  T_tilde = Templates - np.mean(Templates, axis=1)[:, np.newaxis]

  # calculate norm
  T_norm = np.linalg.norm(T_tilde, axis=1)

  # ncc - run through each template
  for k, Tk in enumerate(T_tilde):

    # search image
    for l in range(m-k-1):

      # correlation with norm
      R[l, k] = (Tk / T_norm[k]) @ (T_tilde[k+l+1] / T_norm[k+l+1])

  return R


def tanh_mapping(x, gamma=0.5, lam=2):
  """
  tanh mapping function
  """
  return 0.5 - 0.5 * np.tanh(np.pi * lam * (x - gamma))


def sdm_mapping(sdm):
  """
  mapping function for sdm with tanh and iterative param search
  """

  # lib for otsu threshold
  from skimage import filters
  
  # init
  r = 0
  k = 0.5
  mu = 0.01
  thr_min, thr_max = 0.05, 0.1
  gamma = 0
  S_map = np.zeros(sdm.shape)
  lam = 1

  # iterative algorithm
  while r < thr_min or r > thr_max:

    # get threshold
    gamma = filters.threshold_otsu(sdm[sdm<k])

    # tanh mapping
    S_map = tanh_mapping(sdm, gamma, 1/gamma)

    # recurrence rate
    r = np.mean(S_map)

    # update params
    if r < thr_min:
      k += mu
    else:
      k -= mu

  return S_map


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
  return np.dot(x, eig_vec), eig_val


