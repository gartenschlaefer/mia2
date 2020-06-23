"""
Automatic Harmonic Transcription
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re

import librosa as libr
import librosa.display

from scipy.io import loadmat

from mia2 import *

def plot_chord_mask(chord_mask, chroma_labels, chord_labels, plot_path):
  """
  plot chord mask
  """

  fig, ax = plt.subplots(1,1)

  img = ax.imshow(chord_mask.T, cmap='Greys', aspect='equal')
  plt.ylabel('chroma')
  plt.xlabel('chord')

  # chroma labels
  ax.set_yticks(np.arange(len(chroma_labels)))
  ax.set_yticklabels(chroma_labels)

  # chord labels
  ax.set_xticks(np.arange(len(chord_labels)))
  ax.set_xticklabels(chord_labels)

  plt.xticks(fontsize=10, rotation=90)
  plt.yticks(fontsize=10, rotation=0)

  plt.ylim([-0.5, 11.5])

  # save
  plt.savefig(plot_path + 'chod_mask.png', dpi=150)


def plot_chroma(C, fs, hop, fmin, bins_per_octave, anno=[], xlim=None, plot_path='./'):
  """
  plot whole song
  """
  plt.figure(figsize=(9, 5))
  libr.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=fs, hop_length=hop, x_axis='time', y_axis='chroma', fmin=fmin, bins_per_octave=bins_per_octave)
  plt.colorbar(format='%+2.0f dB')
  #plt.title('Constant-Q power spectrum')

  # add anno
  if anno:
    plot_add_anno(anno, text_height=5, xlim=xlim)

  if xlim is not None:
    plt.xlim(xlim)

  plt.tight_layout()

  # save
  plt.savefig(plot_path + 'chroma.png', dpi=150)


def plot_add_anno(anno, text_height=1, xlim=None):
  """
  adds annotation to plot
  """

  # text bias in height offset
  text_bias = 0

  start_times, end_times, chord_names = anno

  # annotations
  for t_s, t_e, chord_name in zip(start_times, end_times, chord_names):

    # limit anno by time
    if xlim is not None:

      # end time
      if t_s + 1.0 >= xlim[1]:
        return

    # draw vertical lines
    plt.axvline(x=t_s, dashes=(1, 1), color='k')

    # add text
    plt.text(x=t_s, y=text_height + text_bias, s=chord_name, color='k', fontweight='semibold')

    # text bias
    if text_bias >= 6:
      text_bias = -1

    text_bias += 1


def plot_A(A, chord_labels, plot_path, name='A'):
  """
  plot transition probability matrix
  """

  # set up plot
  fig, ax = plt.subplots(1, 1, figsize=(8, 6))

  # plot image
  img = ax.imshow(A, cmap='Greys', aspect='equal')

  # add colorbar
  plt.colorbar(img, ax=ax)

  plt.ylabel('chord')
  plt.xlabel('chord')

  # chroma labels
  ax.set_yticks(np.arange(len(chord_labels)))
  ax.set_yticklabels(chord_labels)

  # chord labels
  ax.set_xticks(np.arange(len(chord_labels)))
  ax.set_xticklabels(chord_labels)

  plt.xticks(fontsize=9, rotation=90)
  plt.yticks(fontsize=9, rotation=0)

  # save
  plt.savefig(plot_path + name + '.png', dpi=150)


def plot_B(B, chord_labels, plot_path, name='B'):
  """
  plot observation probabilities B
  """

  # set up plot
  fig, ax = plt.subplots(1, 1, figsize=(9, 5))

  img = ax.imshow(B, aspect='auto')

  # colorbar
  plt.colorbar(img, ax=ax)

  plt.ylabel('chord')
  plt.xlabel('frames')

  # chroma labels
  ax.set_yticks(np.arange(len(chord_labels)))
  ax.set_yticklabels(chord_labels)

  plt.yticks(fontsize=9, rotation=0)

  # save
  plt.savefig(plot_path + name + '.png', dpi=150)


def plot_viterbi_path(path, t, hop, fs, chord_labels, anno, xlim=None, plot_path='./', name='viterbi_path'):
  """
  plot viterbi path
  """

  # create an image
  path_img = np.zeros(B.shape)

  # create image
  for i, p in enumerate(path):

    # set choosen path to one
    path_img[p, i] = 1

  # extract anno
  start_times, end_times, chord_names = anno

  # annotations
  for t_s, t_e, chord_name in zip(start_times, end_times, chord_names):

    # substituted some stuff
    cn = re.sub(r'(/[\w0-9]+)|(:maj[\w0-9]+)|(:7)', '', chord_name)
    cn = re.sub(r':min', 'm', cn)

    # skip breaks
    if (chord_name == 'N'):
      continue

    # determine position in chromalabels
    p = np.where(np.array(chord_labels) == cn)[0][0]
    #print("chord name: {} new: {} at pos: {} ".format(chord_name, cn, p))

    # mark anno
    path_img[p, int(t_s*fs/hop):int(t_e*fs/hop)] = path_img[p, int(t_s*fs/hop):int(t_e*fs/hop)] * 2 + 2.5


  # set up plot
  fig, ax = plt.subplots(1, 1, figsize=(9, 5))

  # plot
  img = ax.imshow(path_img, aspect='auto', cmap='magma')

  plt.ylabel('chord')
  plt.xlabel('time [s]')

  # chroma labels
  ax.set_yticks(np.arange(len(chord_labels)))
  ax.set_yticklabels(chord_labels)

  ax.set_xticks( np.arange(0, int(t[-1]), 5) * fs / hop )
  ax.set_xticklabels( np.arange(0, int(t[-1]), 5) )

  #plt.xticks(fontsize=9, rotation=90)
  plt.yticks(fontsize=9, rotation=0)

  if xlim is not None:
    plt.xlim(xlim * fs / hop)

  # save
  plt.savefig(plot_path + name + '.png', dpi=150)


if __name__ == "__main__":
  """
  main function
  """

  # plot paths
  plot_path = './ignore/ass9_data/plots/'

  # file paths
  annotation_file = './ignore/ass9_data/Annotations/The_Beatles_Chords/Eight_Days_a_Week.lab'
  a_matrix_mat = './ignore/ass9_data/Files2go/A_Beatles.mat'

  # sound file
  sound_file = './ignore/ass9_data/Files2go/EightDaysAWeek.wav'

  # cqt save path
  c_save_path = './ignore/ass9_data/c.npy'
  c_hat_save_path = './ignore/ass9_data/c_hat.npy'
  fs_save_path = './ignore/ass9_data/fs.npy'
  x_save_path = './ignore/ass9_data/x.npy'

  # extract annotation data: (t_start, t_end, chord)
  anno = extract_lab_file(annotation_file)

  # some params
  hop = 512


  # --
  # naming convention
  # M - length of chroma vector
  # N - number of states, e.g. num of chords to detect
  # K - number of time steps

  # redo chroma calc
  redo = False

  # calc chroma if not already done
  if not os.path.exists(c_save_path) or not os.path.exists(fs_save_path) or not os.path.exists(x_save_path) or redo:

    # read audio
    x, fs = libr.load(sound_file)

    # chromagram c: [K x T]
    c = calc_chroma(x, fs, hop=hop, n_octaves=4, bins_per_octave=36, fmin=librosa.note_to_hz('C3'))

    # remove transient noise
    c_hat = matrix_median(c, n_med=6)

    # save chroma
    np.save(c_save_path, c)
    np.save(c_hat_save_path, c_hat)
    np.save(fs_save_path, fs)
    np.save(x_save_path, x)

  # load
  else:

    # load
    c = np.load(c_save_path)
    c_hat = np.load(c_hat_save_path)
    x = np.load(x_save_path)
    fs = np.load(fs_save_path)


  # create time vector for hop frames
  t = np.arange(0, hop / fs * c.shape[1], hop / fs)

  tm, ts = divmod(t, 60)

  # print some infos
  print("x: ", x.shape), print("fs: {}, time: {}:{}".format(fs, int(tm[-1]), int(ts[-1])))


  # chord prototypes M: [N x M]
  M, chroma_labels, chord_labels = create_chord_mask(maj7=False, g6=False)

  # number of chords
  N = len(chord_labels)


  # --
  # prior probabilities: Pi
  
  # equal prior probabilities
  Pi = 1 / N * np.ones(N)


  # --
  # transition probabilities: A [N x N]

  # heuristic approach
  #A = loadmat(a_matrix_mat)['P']

  # music theoretical approach
  A = get_transition_matrix_circle5ths()


  # --
  # observation probabilities: B: [M x K]

  B = M @ c_hat


  # --
  # Viterbi algorithm search best chord sequence

  from viterbi_path import viterbi_path

  # get best path
  path = viterbi_path(Pi, A, B, scaled=True, ret_loglik=False)


  # --
  # plots

  xlim = (0, 60)
  #xlim = None

  # plot viterbi path
  plot_viterbi_path(path, t, hop, fs, chord_labels, anno, xlim=xlim, plot_path=plot_path, name='viterbi_path')
  
  # plot observation probabilities
  #plot_B(B, chord_labels, plot_path, name='B')

  # plot transition probability matrix A
  #plot_A(A, chord_labels, plot_path, name='A_music')

  # chroma
  plot_chroma(c, fs, hop=512, fmin=librosa.note_to_hz('C3'), bins_per_octave=12, anno=anno, xlim=xlim, plot_path=plot_path)
  plot_chroma(c_hat, fs, hop=512, fmin=librosa.note_to_hz('C3'), bins_per_octave=12, anno=anno, xlim=xlim, plot_path=plot_path)

  # chord mask
  #plot_chord_mask(M, chroma_labels, chord_labels, plot_path)

  # some prints
  print("c: ", c.shape)
  print("M: ", M.shape)
  print("A beetles: ", A.shape)
  print("B beetles: ", B.shape)

  plt.show()



    




