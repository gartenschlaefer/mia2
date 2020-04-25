# --
# library for mia2 - Music Information retrivAl 2

import numpy as np


# Lecture 4:-------------------------------------------------------------------
def time_shift( matrix, shift, axis=1 ):
  shift_matrix = np.roll( matrix, shift, axis )

  assert type( shift) is int, "Shift value is not an integer: %r" % shift

  if shift >= 0: 
    shift_matrix[ : , :shift ] = 0

  elif shift < 0: 
    shift_matrix[ : , shift: ] = 0

  return shift_matrix

def calc_nmf(V, r=7, algorithm='lee', max_iter=100, n_print_dist=10):
  """
  perform a non-negative matrix factorization with selected algorithms
  V: [m x n] = [features x samples] r: num of factorized components
  algortithm:
    - 'lee': Lee and Seung (1999)
  """

  # shape of V: [m features (DFT) x n samples (frames)]
  m, n = V.shape
  
  # right-hand-side matrix 
  H = np.random.rand(r, n)

  # left-hand-side matrix
  W = np.random.rand(m, r)

  # ones matrix
  Ones = np.ones((m, n))

  # iterative update
  for i in range(1, max_iter + 1):

    if algorithm == 'lee':
    
      # update right-hand side matrix
      H = H * ( (W.T @ (V / (W @ H))) / (W.T @ Ones) )

      # update left-hand side matrix
      W = W * ( ((V / (W @ H) @ H.T )) / (Ones @ H.T) )

    if algorithm == 'smaragdis':
      print( 'test' )

    # distance measure
    d = kl_div(V, W @ H)

    # print distance mearure each 
    if not i % n_print_dist or i == max_iter:
      print("iteration: [{}], distance: [{}]".format(i, d))

  return W, H, d


# Lecture 3:-------------------------------------------------------------------
def kl_div(x, y):
  """
  Kullback - Leibler Divergence as distance measure
  """
  return np.linalg.norm(x * np.log(x / y) - x + y)


def get_onset_mat(file_name, var_name='GTF0s'):
  """
  reads a .mat file with midi notes and gives back
  onsets and the corresponding midi notes
  """

  from scipy.io import loadmat

  # read mat files
  mat = loadmat(file_name)

  # get midi notes with funcdamental frequencies
  m = np.round(mat[var_name])

  # gradients of notes
  onsets = np.pad(np.diff(m), ((0, 0), (0, 1)))

  # set onsets to one
  onsets[np.abs(onsets) > 0] = 1

  # get time vector
  t = np.arange(0.023, 0.023+0.01*(m.shape[1]), 0.01);

  return (onsets, m, t)


def non_linear_mapping( u=1, alpha=15, beta=0.5 ):
  """ numpy array, float, float  -> numpy array 
  
  Nonlinear mapping function for the estimation of the fundamental
  distribution. 

  For further information see 'Specmurt Analysis of Polyphonic Music Signals';
  in IEEE Transactions on Audio Speech and language Processing, April 2008 by
  Kameoka,H.; Sagayama S., in particular: Section 4.A.

  As a nonlinear function, a sigmoid function or a hard 
  threshholding can be used:

  u_bar(x)  = u(x) / (1 + exp{-alpha( u(x)/u_max - beta)} ),
  u_max     = max{u(x)} for all x (x -> log-scaled frequency) 

  input params (see equation 13 of the same paper):
  -------------------------------------------------
  @u      ... numpy array containing the powerspectrum, log-frequency  scaled.
  @alpha  ... represents a degree o f fuzziness
  @beta   ... threshhold magnitude paramter, corresponds to the value under 
              which frequency components are assumed to be unwanted.

  output params (see equation 14 of the same paper):
  --------------------------------------------------
  @u_bar  ... estimate of the fundamental frequency distribution

  """
  
  u_max = np.amax( u, axis=0 )
  u_bar =  u / (1 + np.exp( -alpha * (u / u_max - beta )))

  return u_bar

# Lecture 2:-------------------------------------------------------------------
def matrix_otsu_thresh(R, lower_bound=0.5):
  """
  otsu threshold on matrix row, with lower value bound as ignore vals
  """

  from skimage import filters

  # init
  R_bin = np.zeros(R.shape)

  # run through each row
  for row in range(R.shape[0]):

    # valid values
    valid = R[row]>=lower_bound

    # no valid values (needs some values for thresh)
    if np.sum(valid) <= 3:
      continue

    # get otsu thresh
    thresh_row = filters.threshold_otsu(R[row][valid])

    # binarized matrix
    R_bin[row][R[row]>=thresh_row] = 1

  return R_bin


def matrix_median(R, n_med=4):
  """
  median filtering on matrix rows
  """

  from skimage.util import view_as_windows

  # half median size
  n_h = n_med // 2

  # shape
  m, n = R.shape

  # filter matrix to be filtered
  R_fil = view_as_windows(np.pad(R, ((0, 0), (n_h, n_h))), (1, n_med+1), step=1).reshape(m, n, n_med+1)

  return np.median(R_fil, axis=2)


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


# Lecture 1:-------------------------------------------------------------------
def calc_pca(x):
  """
  calculate pca of signal, already ordered, n x m (samples x features)
  """

  # eigen stuff -> already sorted
  eig_val, eig_vec = np.linalg.eig(np.cov(x, rowvar=False))

  # pca transformation
  return np.dot(x, eig_vec), eig_val


# some basics-------------------------------------------------------------------
def custom_stft(x, N=1024, hop=512, norm=True):
  """
  short time fourier transform
  """
  # windowing
  w = np.hanning(N)

  # apply windows
  x_buff = np.multiply(w, buffer(x, N, N-hop))

  # transformation matrix
  H = np.exp(1j * 2 * np.pi / N * np.outer(np.arange(N), np.arange(N)))

  # normalize if asked
  if norm:
    return 2 / N * np.dot(x_buff, H)

  # transformed signal
  return np.dot(x_buff, H)


def buffer(x, n, ol=0):
  """
  buffer function like in matlab
  """

  # number of samples in window
  n = int(n)

  # overlap
  ol = int(ol)

  # hopsize
  hop = n - ol

  # number of windows
  win_num = (len(x) - n) // hop + 1 

  # remaining samples
  r = int(np.remainder(len(x), hop))
  if r:
    win_num += 1;

  # segments
  windows = np.zeros((win_num, n))

  # segmentation
  for wi in range(0, win_num):

    # remainder
    if wi == win_num - 1 and r:
      windows[wi] = np.concatenate((x[wi * hop :], np.zeros(n - len(x[wi * hop :]))))

    # no remainder
    else:
      windows[wi] = x[wi * hop : (wi * hop) + n]

  return windows
