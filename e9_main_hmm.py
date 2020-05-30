"""
Automatic Harmonic Transcription
"""

import numpy as np
import matplotlib.pyplot as plt
import os

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
  plt.figure(2, figsize=(8, 4))
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
  fig, ax = plt.subplots(1,1, figsize=(8, 6))

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
  chroma_save_path = './ignore/ass9_data/chroma.npy'

  # extract annotation data: (t_start, t_end, chord)
  anno = extract_lab_file(annotation_file)


  # --
  # naming convention
  # K - length of chroma vector
  # N - number of states, e.g. num of chords to detect
  # T - number of time steps


  # calc chroma if not already done
  if not os.path.exists(chroma_save_path):

    # read audio
    x, fs = libr.load(sound_file)

    # print some infos
    print("x: ", x.shape), print("fs: ", fs)

    # chromagram c: [K x T]
    c = calc_chroma(x, fs, hop=512, n_octaves=4, bins_per_octave=36, fmin=librosa.note_to_hz('C3'))

    # save chroma
    np.save(chroma_save_path, c)

  # load
  else:

    # load
    c = np.load(chroma_save_path)



  # chord prototypes M: [N x K]
  M, chroma_labels, chord_labels = create_chord_mask(maj7=False, g6=False)

  # number of chords
  N = len(chord_labels)


  # --
  # prior probabilities
  
  # equal start probability
  p0 = 1 / N * np.ones(N)

  # TODO: with basic key


  # --
  # transition probabilities: A [N x N]

  # heuristic approach
  #A = loadmat(a_matrix_mat)['P']

  # music theoretical approach
  A = get_transition_matrix_circle5ths()




  # --
  # observation probabilities: B: [N x K]

  # TODO: check this wrong dimension
  B = M @ c



  # --
  # Viterbi algorithm search best chord sequence



  # --
  # plots

  # plot transition probability matrix A
  #plot_A(A, chord_labels, plot_path, name='A_music')

  # chroma
  #plot_chroma(c, fs, hop=512, fmin=librosa.note_to_hz('C3'), bins_per_octave=12, anno=anno, xlim=(0, 30), plot_path)

  # chord mask
  #plot_chord_mask(M, chroma_labels, chord_labels, plot_path)

  # some prints
  print("c: ", c.shape)
  print("M: ", M.shape)
  #print("A beetles: ", A)
  print("A beetles: ", A.shape)
  #print("B beetles: ", B)
  print("B beetles: ", B.shape)

  plt.show()



    




