"""
Automatic Harmonic Transcription
"""

import numpy as np
import matplotlib.pyplot as plt

import librosa as libr

from mia2 import *


def cqt_approach():
  """
  cqt approach
  """

  # hop size
  hop = 256

  # start note
  start_note = libr.note_to_hz( 'C2' )

  # cqt
  cqt = libr.cqt( x, sr=fs, hop_length=hop, fmin=start_note, n_bins=48, bins_per_octave=12 )

  # plot cqt
  plt.figure(), plt.imshow(np.abs(cqt), aspect='auto')


if __name__ == "__main__":
  """
  main function
  """

  # plot paths
  plot_path = 'ignore/ass8_data/plots/'

  # file path
  file_path = 'ignore/ass8_data/01-AchGottundHerr.wav'

  # read audio
  x, fs = libr.load( file_path )

  # print some infos
  print("x: ", x.shape), print("fs: ", fs)

  # window size
  N = 2048


  # create half tone filter bank
  Hp = create_half_tone_filterbank(N, fs, midi_start_note=43, num_oct=4)

  # print shape of Hp filter bank
  print("Hp: ", Hp.shape)

  # plot Hp
  plt.figure(), plt.imshow(Hp, aspect='auto')


  # --
  # cqt

  #cqt_approach():


  # show plots
  plt.show()

    




