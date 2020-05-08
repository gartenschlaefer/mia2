# --
# library for mia2 - Music Information retrivAl 2

import numpy as np

# Lecture 7:-------------------------------------------------------------------

def calc_fisher_ratio(x, y):
  """
  calculate the fisher ratio of each feature and each class
  x: [n x m] n samples, m features
  """

  #  n samples, m features
  n, m = x.shape

  # amount of classes
  labels = np.unique(y)
  n_classes = len(labels)

  # compare labels
  compare_label = []

  # get all labels to compare
  for i in range(n_classes - 1):
    for i_s in range(i + 1, n_classes):

      # append to label compare list
      compare_label.append(labels[i] + ' - ' + labels[i_s])

  # init ratio
  r = np.zeros((m, len(compare_label)))

  # all features
  for j in range(m):

    # comparison class
    c = 0

    # all class compares
    for i in range(n_classes - 1):

      for i_s in range(i + 1, n_classes):

        # calculate fisher ration 
        r[j, c] = (np.mean(x[y==labels[i], j]) - np.mean(x[y==labels[i_s], j]))**2 / (np.var(x[y==labels[i], j]) + np.var(x[y==labels[i_s], j]) )
        c += 1
  
  return r, compare_label


def calc_dp(x, y):
  """
  calculate discriminance potential
  """

  # calculate scatter matrices
  Sw, Sb, cov_k, label_list = calc_class_scatter_matrices(x, y)

  return np.trace(Sw) / np.trace(Sb)


def feature_filter(x, y):
  """
  feature filter uses the filter approach to reduce feature dimensions
  x: [n x m] n samples, m features
  """

  # get shape of things
  n, m = x.shape

  # TODO: implementation

  return x, m


def feature_wrapper(x, y):
  """
  feature wrapper uses the wrapper approach to reduce feature dimensions
  x: [n x m] n samples, m features
  """
  # get shape of things
  n, m = x.shape

  # TODO: implementation

  return x, m


# Lecture 6:-------------------------------------------------------------------

def lda_classify(x, w, bias, label_list):
  """
  classification with lda classifier using weights and bias
  return predicted classes y_hat
  transforms data: 
  [
    x_h = w.T @ x
  ]
  and classifies it
  """

  # TODO: implementation

  return None


def calc_class_scatter_matrices(x, y):
  """
  calculates the within-class scatter matrix Sw and
  between-class scatter matrix Sb and Covariance matrix Cov_k
  """

  # n samples, m features
  n, m = x.shape

  # labels and classes
  labels = np.unique(y)
  n_classes = len(labels)

  # overall mean [m]
  mu = np.mean(x, axis=0)

  # init statistics
  p_k, mu_k, cov_k = np.zeros(n_classes), np.zeros((n_classes, m)), np.zeros((n_classes, m, m))

  # init label list
  label_list = []

  # calculate statistics from samples for further processing
  for k, label in enumerate(labels):

    # append label
    label_list.append(label)

    # get class samples
    class_samples = x[y==label, :]

    # class occurrence probability [k]
    p_k[k] = len(class_samples) / n

    # mean vector of classes [k x m]
    mu_k[k] = np.mean(class_samples, axis=0)

    # covariance matrix of classes [k x m x m]
    cov_k[k] = np.cov(class_samples, rowvar=False)


  # calculate between class scatter matrix S_b [m x m]
  Sb = np.einsum('k, km, kn -> mn', p_k, mu_k-mu, mu_k-mu)

  # calculate within class scatter matrix S_w [m x m]
  Sw = np.einsum('k, kmn -> mn', p_k, cov_k)

  return Sw, Sb, cov_k, label_list


def train_lda_classifier(x, y, method='class_independent', n_lda_dim=1):
  """
  train lda classifier, extract weights and bias vectors x:[n samples x m features]
  return weights, biases, transformed data and label list
  """

  # n samples, m features
  n, m = x.shape

  # calculate scatter matrices
  Sw, Sb, cov_k, label_list = calc_class_scatter_matrices(x, y)

  # number of classes
  n_classes = len(label_list)

  # class independent method - standard: use S_w
  if method == 'class_independent':

    # compute eigenvector
    eig_val, eig_vec = np.linalg.eig(np.linalg.inv(Sw) @  Sb)
    
    # real valued eigenvals [k-1 x m]
    w = eig_vec[:n_classes-1, :]

    # transformed data [k-1 x n] = [k-1 x m] @ [m x n]
    x_h = w @ x.T

    # bias [k-1]
    bias = np.mean(x_h, axis=1)


  # class dependent use covariance instead of S_w
  elif method == 'class_dependent':

    # init
    w = np.zeros((n_classes, m, n_lda_dim))
    bias = np.zeros(n_classes)
    x_h = np.zeros((n, n_lda_dim))

    # run through all classes
    for k in range(n_classes):

      # compute eigenvector
      eig_val, eig_vec = np.linalg.eig(np.linalg.inv(cov_k[k]) @  Sb)

      # use first eigenvector
      w[k] = eig_vec[:, :n_lda_dim].real

      # transformed data
      x_h[y==label_list[k]] = (w[k].T @ x[y==label_list[k]].T).T

      # bias
      bias[k] = np.mean(x_h[y==label_list[k]])

  return w, bias, x_h, label_list


def compute_lambda( W, H, T, M, N ):
  
  # First init. of Lambda matrix
  Lambda = np.zeros( (M, N) )

  for t in range( T ):
    Lambda += W[ :, :, t ] @ time_shift( H, t, axis=1 )

  return Lambda


# Lecture 4:-------------------------------------------------------------------
def time_shift( matrix, shift, axis=1 ):
  shift_matrix = np.roll( matrix, shift, axis )

  assert type( shift ) is int, "Shift value is not an integer: %r" % shift

  if shift >= 0: 
    shift_matrix[ : , :shift ] = 0

  elif shift < 0: 
    shift_matrix[ : , shift: ] = 0

  return shift_matrix

def calc_nmf(V, R=7, T=10, algorithm='lee', max_iter=100, n_print_dist=10):
  """
  perform a non-negative matrix factorization with selected algorithms
  V: [m x n] = [features x samples] r: num of factorized components
  algorithm:
    - 'lee': Lee and Seung (1999)
  """

  # shape of V: [m features (DFT) x n samples (frames)]
  M, N = V.shape
  
  # right-hand-side matrix 
  H = np.random.rand( R, N )

  # left-hand-side matrix, depending on the algorithm
  W = np.random.rand( M, R )
  
  # smaragdis algorithm init wt and ht matrix
  if algorithm == 'smaragdis':

    # init W matrix with time
    W = np.ones( (M, R, T) ) * 1e-4

    # Ht for collecting H over t
    H_t = np.zeros( (R, N, T) )

    # lambda matrix for nmf-deconvolution
    Lambda = compute_lambda( W, H, T, M, N )

  # ones matrix
  Ones = np.ones( (M, N) )

  # iterative update
  for i in range(1, max_iter + 1):

    if algorithm == 'lee':
    
      # update right-hand side matrix
      H = H * ( (W.T @ (V / (W @ H))) / (W.T @ Ones) )

      # update left-hand side matrix
      W = W * ( ((V / (W @ H) @ H.T )) / (Ones @ H.T) )

      # distance measure
      d = kl_div(V, W @ H)

    if algorithm == 'smaragdis':

      # run through each time step for H
      for t in range( T ):

        # collect time steps
        H_t[:, :, t] = ( ( W[:, :, t].T @ time_shift(V / Lambda, -1*t) ) / ( W[:, :, t].T @ Ones) )

      # update right-hand side matrix
      H = H * np.mean( H_t, axis=2 )

      # run through each time step for W
      for t in range( T ):

        # update left-hand side matrix
        W[:, :, t] = W[:, :, t] * ( ( (V / Lambda) @ time_shift( H, t ).T ) / ( Ones @ time_shift( H, t ).T ) )

        # normalize W
        W[:, :, t] = W[:, :, t] / np.linalg.norm(W[:, :, t])

      # Update Lambda matrix for the next iteration
      Lambda = compute_lambda( W, H, T, M, N )

      # distance measure
      d = kl_div( V, Lambda )

    # print distance measure each 
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

  # get midi notes with fundamental frequencies
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
  thresholding can be used:

  u_bar(x)  = u(x) / (1 + exp{-alpha( u(x)/u_max - beta)} ),
  u_max     = max{u(x)} for all x (x -> log-scaled frequency) 

  input params (see equation 13 of the same paper):
  -------------------------------------------------
  @u      ... numpy array containing the power-spectrum, log-frequency  scaled.
  @alpha  ... represents a degree o f fuzziness
  @beta   ... threshold magnitude parameter, corresponds to the value under 
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
  calculate pca of signal, already ordered, m x n (samples x features)
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
