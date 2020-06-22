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
  midi_start_note=43

  # window size
  N = 2048

  # create half tone filter bank
  Hp, chroma_labels = create_half_tone_filterbank(N, fs, midi_start_note=midi_start_note, num_oct=4)

  # calculate pitches
  c_half = calc_chroma_halftone(x, Hp, N)

  # chroma with cqt with tuning
  c_cqt = calc_chroma(x, fs, hop=N//2, n_octaves=5, bins_per_octave=36, fmin=libr.note_to_hz(start_note))

  # calc CRP
  c_crp = calc_crp( x, Hp, N, midi_start_note=midi_start_note)

  # choose chroma for further processing
  chroma = c_cqt

  # remove transient noise
  c_hat = matrix_median(chroma, n_med=6)

  # beat snychron smoothing of chromagram 
  # get onsets for smoothings

  # --
  # some plots

  # pitches
  fig, ax = plt.subplots(3, 1)
  plt.subplots_adjust(hspace = 0.5)
  ax[0].imshow(c_half, aspect='auto'), ax[0].set_title("chroma halftone")
  ax[1].imshow(c_cqt, aspect='auto'), ax[1].set_title("chroma cqt")
  im = ax[2].imshow(c_crp, aspect='auto')
  ax[2].set_title("chroma crp")
  #plt.colorbar(im, ax=ax[2])

  #plt.figure(), plt.title("c pitch"), plt.imshow(c_pitch, aspect='auto')
  #plt.figure(), plt.title("chroma"), plt.imshow(chroma, aspect='auto')

  #fig, ax = plt.subplots(2, 1)
  #fig.tight_layout()
  #plt.subplots_adjust(hspace = 0.5)
  #ax[0].imshow(chroma, aspect='auto'), ax[0].set_title("chroma tuned"), ax[1].imshow(c_hat, aspect='auto'), ax[1].set_title("chroma without transient noise")

  print("chroma:", chroma_labels)
  # plot Hp
  #plt.figure(), plt.imshow(Hp, aspect='auto')

  # show plots
  plt.show()

    




