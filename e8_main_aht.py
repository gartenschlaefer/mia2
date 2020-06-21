"""
Automatic Harmonic Transcription
"""

import numpy as np
import matplotlib.pyplot as plt

import librosa as libr

from mia2 import *


def cqt_approach(x, fs):
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

  # --
  # cqt

  #cqt_approach()


  # --
  # CRP

  # start note for chroma algorithm G2:43
  start_note = 'G2'

  # window size
  N = 2048

  # create half tone filter bank
  Hp = create_half_tone_filterbank(N, fs, midi_start_note=43, num_oct=4)

  # calculate pitches
  c_pitch = calc_pitch_gram(x, Hp, N)

  # comapre with chroma
  chroma = calc_chroma(x, fs, hop=N//2, n_octaves=5, bins_per_octave=36, fmin=libr.note_to_hz(start_note))



  # --
  # some plots

  # pitches
  plt.figure(), plt.title("c pitch"), plt.imshow(c_pitch, aspect='auto')
  plt.figure(), plt.title("chroma"), plt.imshow(chroma, aspect='auto')

  # plot Hp
  #plt.figure(), plt.imshow(Hp, aspect='auto')

  # show plots
  plt.show()

    




