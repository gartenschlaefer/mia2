# --
# library for mia2 - Music Information retrivAl 2

import numpy as np


# Lecture 10:-------------------------------------------------------------------
def label_to_index(y, labels):
  """
  get label and produce int index
  """

  for i, label in enumerate(labels):

    # replace label with index
    y[np.where(y == label)] = i

  return y.astype(int)


# Lecture 9:-------------------------------------------------------------------

def smoothed_beat_chroma(x, fs, hop=1024, n_octaves=4, fmin=65.40639132514966, n_med=6):
  """
  smoothed beat chromagram
  """

  import librosa

  # chroma with cqt with tuning
  c = calc_chroma(x, fs, hop=hop, n_octaves=n_octaves, bins_per_octave=36, fmin=fmin)

  # remove transient noise
  c = matrix_median(c, n_med=n_med)

  # beat snychron smoothing of chromagram 
  tempo, beats = librosa.beat.beat_track(x, sr=fs, hop_length=hop)

  # feature averaging over beats
  c = frame_filter(c, beats, filter_type='mean')

  return c, beats


def get_transition_matrix_circle5ths(gamma=1):
  """
  get transition matrix with music theoretical approach with circle of fifths
  """

  # chroma labels
  chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

  # chord labels
  chord_labels = chroma_labels + [c + "m" for c in chroma_labels]

  # circle steps with minor parallel in between
  circle_steps = ['C', 'Am', 'F', 'Dm', 'A#', 'Gm', 'D#', 'Cm', 'G#', 'Fm', 'C#', 'A#m', 'F#', 'D#m', 'B', 'G#m', 'E', 'C#m', 'A', 'F#m', 'D', 'Bm', 'G', 'Em']

  # init A
  A = np.zeros((len(chord_labels), len(chord_labels)))

  # sophisticated algorithm
  for i, cl in enumerate(chord_labels):

    # roll to base
    cs_roll = np.roll(circle_steps, -np.where(np.array(circle_steps) == cl)[0][0])

    # get position in chord labels
    circle_chord_pos = []

    # calculate distance
    d = np.zeros(len(chord_labels))

    for j, cs in enumerate(cs_roll):

      # position of chord
      circle_chord_pos = np.where(np.array(chord_labels) == cs)[0][0]

      # if it is more than 12 go other direction
      if j > 12:
        dj = 12 - j % 12
      else:
        dj = j

      # get distance
      d[circle_chord_pos] = dj

    # update A
    A[i] = d

  # do some scaling
  A = (12 - A + gamma) / (144 + 24 * gamma)

  return A


def create_chord_mask(maj7=False, g6=False, start_note='C'):
  """
  create a chord mask
  """

  # dur
  dur_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

  # mol
  mol_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])

  # maj7
  maj7_template = np.array([0.75, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 0, 0, 1.2])

  # 6
  g6_template = np.array([0.75, 0, 0, 0, 0.75, 0, 0, 0.75, 0, 1, 0, 0])

  # chroma labels
  #chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  chroma_labels = list(get_chroma_labels(start_note=start_note))

  # chord labels
  chord_labels = chroma_labels + [c + "m" for c in chroma_labels]

  # templates concatenated
  chord_templates = np.array([dur_template, mol_template])

  # append maj7
  if maj7 ==  True:
    chord_templates = np.vstack((chord_templates, maj7_template))
    chord_labels = chord_labels + [c + "maj7" for c in chroma_labels]

  # append g6
  if g6 == True:
    chord_templates = np.vstack((chord_templates, g6_template))
    chord_labels = chord_labels + [c + "6" for c in chroma_labels]

  # init mask
  chord_mask = np.empty((0, 12), int)

  # go through all templates
  for chord_template in chord_templates:
    
    # all chroma values
    for c in range(12):

      # add to events
      chord_mask = np.vstack((chord_mask, np.roll(chord_template, c)))

  return chord_mask, chroma_labels, chord_labels


def extract_lab_file(annotation_file):
  """
  extract data from lab file (t_start, t_end, Akkord)
  """

  # create anno lists
  t_start_list = []
  t_end_list = []
  akkord_list = []

  # open file
  with open(annotation_file, 'r') as file:

    # read line
    for line in file:

      # strip line of line breaks
      line = line.strip()

      # split entries
      t_s, t_e, akk = line.split()

      # append to lists
      t_start_list.append(float(t_s))
      t_end_list.append(float(t_e))
      akkord_list.append(akk)

  return t_start_list, t_end_list, akkord_list


# Lecture 8:-------------------------------------------------------------------
def harmonic_change_detection(c_proj):
  """
  harmonic change detection function on tonal centroid projection
  """
  
  # difs
  d = np.roll(np.roll(c_proj, -2, axis=1) - c_proj, 1, axis=1)

  # cleanup
  d[:, 0] = 0
  d[:, -1] = 0

  # distance calculation
  hcdf = np.linalg.norm(d, axis=0)

  return hcdf


def tonal_centroid_proj(c, r=[1, 1, 0.5]):
  """
  tonal centroid projection
  c: chroma with shape [m x n]
  m: features
  n: samples
  """

  # shape of things
  m, n = c.shape

  # length vector of chroma
  l = np.arange(m)

  # centroid projection matrix
  phi = np.vstack(( r[0] * np.sin(7 * np.pi / 6 * l),
                    r[0] * np.cos(7 * np.pi / 6 * l),
                    r[1] * np.sin(3 * np.pi / 2 * l),
                    r[1] * np.cos(3 * np.pi / 2 * l),
                    r[2] * np.sin(2 * np.pi / 3 * l),
                    r[2] * np.cos(2 * np.pi / 3 * l) ))

  # projection
  xi = (phi @ c) / np.linalg.norm(c, ord=1, axis=0)

  return xi


def get_chroma_labels(start_note='C'):

  # chroma labels depending on start note
  chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

  return np.roll(chroma_labels, -np.where(np.array(chroma_labels) == start_note)[0][0])
  

def f_to_midi_scale(f):
  """
  convert frequency to midi scale
  """
  return 12 * np.log2( f / 440 ) + 69


def create_half_tone_filterbank(N, fs, midi_start_note=43, num_oct=4):
  """
  create half-tone filterbank
  """
  import librosa

  # midi notes
  p = np.arange(midi_start_note, midi_start_note + 12 * num_oct)

  # midi notes of discrete DFT-bins
  p_fk = np.insert( f_to_midi_scale(np.arange(1, N/2) * fs / N), 0, 0)

  # differences
  d = np.abs(p[:, np.newaxis] - p_fk)

  # half-tone filterbank
  Hp = 0.5 * np.tanh(np.pi * (1 - 2 * d)) + 0.5

  return Hp, get_chroma_labels(start_note=librosa.midi_to_note(midi_start_note, octave=False))


def calc_chroma_halftone( x, hp, n, bins_per_octave=12 ):
  """ 
  - Needs buffered input signal.
  - Length of a segment.
  - perform FFT and take the absolute value 
  fft( signal[ 1:lenght(signal) / 2 + 1, : ] ) 

  - Only half of the blocks are needed due to the symmetry of the fft
  of a real valued signal.

  The pitchgramm itself is the transformed signal multiplicated 
  with the halftone filter bank.

  """

  # windowing
  x_buff_win = buffer( x, n, ol=n//2 ) * np.hanning( n )

  # fft
  X_buff_win = np.abs( np.fft.fft( x_buff_win, n ))[:, :n//2]

  # filter [m x n]
  p = hp @ X_buff_win.T

  # sum over octaves
  return np.sum(np.abs(buffer2D(p, bins_per_octave)), axis=0)


def calc_crp( x, hp, n, midi_start_note=43, bins_per_octave=12, num_octaves=4, m=120, n_crp=55):
  """ 
  calculate CRP (chroma DCT-reduced log pitch)
  """

  from scipy.fftpack import dct, idct

  # windowing
  x_buff_win = buffer( x, n, ol=n//2 ) * np.hanning( n )

  # fft
  X_buff_win = np.abs( np.fft.fft( x_buff_win, n ))[:, :n//2]

  # filter [m x n]
  pitch = (hp @ X_buff_win.T)**2

  # pitch representation
  v = 1e-8 * np.ones((m, pitch.shape[1]))

  # determine midi end note
  midi_end = midi_start_note + num_octaves * bins_per_octave

  # get pitches to corresponding positions
  v[midi_start_note:midi_end, :] = pitch

  # norm
  v = v / np.linalg.norm(v, ord=2)

  # log
  v_log = np.log(100 * v + 1)

  # dct
  v_dct = dct(v_log, axis=1)

  # set to zero
  v_dct[:m-n_crp, :] = 0

  # norm
  v_dct = v_dct / np.linalg.norm(v_dct, ord=2)

  # lift
  lift = 1 / 100 * ( np.exp(idct(v_dct, axis=1)) - 1 )

  # sum octaves
  crp = np.sum(np.abs(buffer2D(np.abs(lift), bins_per_octave)), axis=0)

  # norm
  crp = crp / np.linalg.norm(crp, ord=2)

  # roll to correct
  crp = np.roll(crp, 5, axis=0)

  return crp


# Lecture 7:-------------------------------------------------------------------
def calc_dp(x, y):
  """
  calculate discriminance potential of lda transformed data
  x: [n x m] n-samples, m-features
  """

  # transform data points with lda
  w, bias, x_h, label_list = train_lda_classifier(x, y)

  # calculate scatter matrices
  Sw, Sb, cov_k, label_list = calc_class_scatter_matrices(x_h.T, y)

  # return dp
  return np.trace(Sb) / np.trace(Sw)


def SFS_Search(x, y, start_features=None, depth=None):
  """
  Sequential forward search
  """

  # get shape of things: n samples, m features
  n, m = x.shape

  # limit for search
  if depth is None:
    depth = m

  # start with zero set [m x n]
  x_h = np.empty(shape=(0, n), dtype=x.dtype)

  # dp_list, actual indices list
  dp_m, act_mi = np.array([]), []

  # selected start features if None -> empty set
  if start_features is not None:

    # start with selected feature set
    x_h = x[:, start_features].T

    # actual indices in x_h
    act_mi = start_features


  # TODO: include stop condition
  for r in range(depth):

    # determine excluded feature indices
    exc_mi = np.delete(np.arange(m), act_mi)

    # cost array
    J = np.zeros(len(exc_mi)) 

    # add excluded mi one by one
    for i, emi in enumerate(exc_mi):

      # append feature for trial
      x_trial = np.vstack((x_h, x[:, emi]))

      # calculate cost
      J[i] = calc_dp(x_trial.T, y)

      #print("added feature index: {} with dp: {}".format(emi, J[i]))

    # determine best contribution feature
    best_mi = exc_mi[np.argmax(J)]

    # append best feature
    x_h = np.vstack((x_h, x[:, best_mi]))

    # actual feature index added
    act_mi = np.append(act_mi, best_mi).astype(int)
    
    # discriminance potential for set of features
    #dp_m.append(J[np.argmax(J)])
    dp_m = np.append(dp_m, J[np.argmax(J)])

    # print message
    print("total feat: {}, best feature index: {} with dp: {}".format(r+1, act_mi[r], dp_m[r]))

  return x_h.T, act_mi, dp_m


def SBS_Search(x, y, start_features=None, depth=None):
  """
  Sequential backward search
  """

  # get shape of things: n samples, m features
  n, m = x.shape

  # limit for search
  if depth is None:
    depth = m // 2

  # start with full set [m x n]
  x_h = x

  # dp_list, actual indices list
  dp_m, act_mi = np.array([]), np.arange(m)

  # selected start features if None -> empty set
  if start_features is not None:

    # start with selected feature set
    x_h = x[:, start_features]

    # actual indices in x_h
    act_mi = start_features


  # TODO: include stop condition
  for r in range(depth):

    # cost array
    J = np.zeros(len(act_mi))

    # add excluded mi one by one
    for i, mi in enumerate(act_mi):

      # remove index
      x_trial = np.delete(x_h, i, axis=1)

      # calculate cost
      J[i] = calc_dp(x_trial, y)

      #print("deleted feature index: {} with dp: {}".format(mi, J[i]))

    # determine worst contribution feature
    worst_mi = act_mi[np.argmin(J)]

    # delete worst feature
    x_h = np.delete(x_h, np.argmax(J), axis=1)
    act_mi = np.delete(act_mi, np.argmax(J))

    # discriminance potential for set of features
    dp_m = np.append(dp_m, J[np.argmax(J)])

    # print message
    print("total feat: {}, worst feature index: {} with dp: {}".format(m-r+1, worst_mi, dp_m[r]))

  return x_h, act_mi, dp_m


def LRS_search(x, y, L, R, max_it=1):
  """
  Plus L - R selection
  """

  # TODO: add stopping condition

  # dp score
  dp_score = np.array([])

  # inti actual feature indices
  act_mi = None

  lr_labels = np.array([])

  # run trough all iterations
  for it in range(max_it):

    # start with backward search
    if R > L:

      # backward search R times
      x_h, act_mi, dp_m1 = SBS_Search(x, y, start_features=act_mi, depth=R)

      # forward search L times
      x_h, act_mi, dp_m2 = SFS_Search(x, y, start_features=act_mi, depth=L)

      # add labels
      lr_labels = np.concatenate((lr_labels, np.ones(R), np.zeros(L)))


    # start with forward search
    else:

      # forward search L times
      x_h, act_mi, dp_m1 = SFS_Search(x, y, start_features=act_mi, depth=L)

      # backward search R times
      x_h, act_mi, dp_m2 = SBS_Search(x, y, start_features=act_mi, depth=R)

      # add labels
      lr_labels = np.concatenate((lr_labels, np.zeros(L), np.ones(R)))

    # append to score
    dp_score = np.concatenate((dp_score, dp_m1, dp_m2))

  return x_h, act_mi, dp_score, lr_labels


def feature_filter(x, y, algorithm='SFS', L=2, R=1, max_it=2):
  """
  feature filter uses the filter approach to reduce feature dimensions
  x: [n x m] n samples, m features
  choose algorithm: 
    sfs - forward search
    sbs - backward search
    lrs - left/right search
  """

  # get shape of things
  n, m = x.shape

  # labels for search
  lr_labels = None

  # forward search
  if algorithm == 'SFS':
    x_h, act_mi, dp_m = SFS_Search(x, y, start_features=None, depth=None)

  # backward search
  elif algorithm == 'SBS':
    x_h, act_mi, dp_m = SBS_Search(x, y, start_features=None, depth=m-1)

  # left, right search
  else:
    x_h, act_mi, dp_m, lr_labels = LRS_search(x, y, L=L, R=R, max_it=max_it)
  
  return x_h, act_mi, dp_m, lr_labels


def feature_wrapper(x, y):
  """
  feature wrapper uses the wrapper approach to reduce feature dimensions
  x: [n x m] n samples, m features
  """
  # get shape of things
  n, m = x.shape

  # TODO: implementation
  act_mi = np.arange(m)

  return x, act_mi


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
  pass

  return None


def calc_class_scatter_matrices(x, y):
  """
  calculates the within-class scatter matrix Sw and
  between-class scatter matrix Sb and Covariance matrix Cov_k
  """

  # n samples, m features
  n, m = x.shape
  
  # labels and classes
  labels = np.unique( y )

  n_classes = len( labels )

  # overall mean [m]
  mu = np.mean( x, axis=0 )

  # init statistics
  p_k, mu_k, cov_k = np.zeros(n_classes), np.zeros((n_classes, m)), np.zeros((n_classes, m, m))

  # init label list
  label_list = []

  # calculate statistics from samples for further processing
  for k, label in enumerate( labels ):

    # append label
    label_list.append( label )

    # get class samples
    class_samples = x[ y==label, : ]

    # class occurrence probability [k]
    p_k[ k ] = len(class_samples) / n

    # mean vector of classes [k x m]
    mu_k[ k ] = np.mean( class_samples, axis=0 )

    # covariance matrix of classes [k x m x m]
    cov_k[ k ] = np.cov( class_samples, rowvar=False )

  # calculate between class scatter matrix S_b [m x m]
  Sb = np.einsum('k, km, kn -> mn', p_k, mu_k-mu, mu_k-mu)

  # calculate within class scatter matrix S_w [m x m]
  Sw = np.einsum('k, kmn -> mn', p_k, cov_k)

  return Sw, Sb, cov_k, mu_k, label_list


def train_lda_classifier(x, y, method='class_independent', n_lda_dim=1):
  """
  train lda classifier, extract weights and bias vectors x:[n samples x m features]
  return weights, biases, transformed data and label list
  """

  # n samples, m features
  n, m = x.shape

  # calculate scatter matrices
  Sw, Sb, cov_k, mu_k, label_list = calc_class_scatter_matrices(x, y)
  
  # number of classes
  n_classes = len( label_list )

  # class independent method - standard: use S_w
  if method == 'class_independent':

    # compute eigenvector
    eig_val, eig_vec = np.linalg.eig( np.linalg.inv(Sw) @  Sb ) 

    # real valued eigenvals [m x k-1]
    w = eig_vec[ : , : n_classes - 1 ].real

    # transformed data [k-1 x n] = [k-1 x m] @ [m x n]
    x_h = w.T @ x.T

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
      eig_val, eig_vec = np.linalg.eig( np.linalg.inv(cov_k[k]) @ Sb )

      # use first eigenvector
      w[ k ] = eig_vec[ :, : n_lda_dim ].real

      # transformed data
      x_h[ y == label_list[k] ] = ( w[k].T @ x[ y == label_list[k] ].T ).T

      # bias
      bias[k] = np.mean(x_h[y==label_list[k]])
  
  mu_k_h = w.T @ mu_k.T

  return w, bias, x_h, mu_k_h, label_list


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

def frame_filter(feature, frames, filter_type='median', norm=False):
  """
  Filtering of consecutive frames defined by frames, median or mean filter
  """

  # init
  m_feature = np.zeros((feature.shape[0], len(frames)))

  # for each frame
  for i, frame in enumerate(frames):

    # stop filtering
    if i == len(frames) - 1:
      end_frame = -1

    else:
      end_frame = frames[i+1]

    # average filter
    if filter_type == 'mean':
      m_feature[:, i] = np.mean(feature[:, frame:end_frame], axis=1)

    # median filter
    else:
      m_feature[:, i] = np.median(feature[:, frame:end_frame], axis=1)

  # normalize
  if norm:
    #print(np.linalg.norm(m_feature, ord=np.inf))
    m_feature = m_feature / np.linalg.norm(m_feature, ord=np.inf)

  return m_feature


def calc_pca(x):
  """
  calculate pca of signal, already ordered, m x n (samples x features)
  """

  # eigen stuff -> already sorted
  eig_val, eig_vec = np.linalg.eig(np.cov(x, rowvar=False))

  # pca transformation
  return np.dot(x, eig_vec), eig_val


# some basics-------------------------------------------------------------------
def calc_dct(x, N):
  """
  discrete cosine transform
  """
  
  # transformation matrix
  H = np.cos(np.pi / N * np.outer((np.arange(N) + 0.5), np.arange(N)))

  # transformed signal
  return np.dot(x, H)


def calc_chroma(x, fs, hop=512, n_octaves=5, bins_per_octave=36, fmin=65.40639132514966):
  """
  calculate chroma values with constant q-transfrom and tuning of the HPCP
  """

  import librosa

  # ctq
  C = np.abs(librosa.core.cqt(x, sr=fs, hop_length=hop, fmin=fmin, n_bins=bins_per_octave * n_octaves, bins_per_octave=bins_per_octave, tuning=0.0, filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect', res_type=None))

  # calculate HPCP
  hpcp = HPCP(C, n_octaves, bins_per_octave=bins_per_octave)

  # make a histogram of tuning bins
  hist_hpcp = histogram_HPCP(hpcp, bins_per_octave)

  # tuning
  tuned_hpcp = np.roll(hpcp, np.argmax(hist_hpcp), axis=0)

  return filter_HPCP_to_Chroma(tuned_hpcp, bins_per_octave, filter_type='median')


def filter_HPCP_to_Chroma(tuned_hpcp, bins_per_octave, filter_type='mean'):
  """
  filter hpcp bins per chroma to a single chroma value, mean and median filters are possible
  """
  if filter_type == 'mean':
    chroma = np.mean(np.abs(buffer2D(tuned_hpcp, bins_per_octave // 12)), axis=1)

  else:
    chroma = np.median(np.abs(buffer2D(tuned_hpcp, bins_per_octave // 12)), axis=1)

  return chroma


def histogram_HPCP(hpcp, bins_per_octave):
  """
  create histogram of tuning bins over all chroma and frames
  """
  return np.sum(np.sum(np.abs(buffer2D(hpcp, bins_per_octave // 12)), axis=0), axis=1)


def HPCP(C, n_octaves, bins_per_octave=12):
  """
  Harmonic Pitch Class Profile calculated from cqt C
  """
  return np.sum(np.abs(buffer2D(C, bins_per_octave)), axis=0)


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


def buffer2D(X, n, ol=0):
  """
  buffer function like in matlab but with 2D
  """

  # number of samples in window
  n = int(n)

  # overlap
  ol = int(ol)

  # hopsize
  hop = n - ol

  # number of windows
  win_num = (X.shape[0] - n) // hop + 1 

  # remaining samples
  r = int(np.remainder(X.shape[0], hop))
  if r:
    win_num += 1;

  # segments
  windows = np.zeros((win_num, n, X.shape[1]), dtype=complex)

  # segmentation
  for wi in range(0, win_num):

    # remainder
    if wi == win_num - 1 and r:
      windows[wi] = np.concatenate((X[wi * hop :], np.zeros((n - X[wi * hop :].shape[0], X.shape[1]))))

    # no remainder
    else:
      windows[wi, :] = X[wi * hop : (wi * hop) + n, :]

  return windows
