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


def Halftone(Hp, plot_path='./', name='chroma_types'):
  """
  halftone filterbank
  """
  plt.figure()
  plt.imshow(Hp, aspect='auto')
  plt.ylabel("midi notes")
  plt.xlabel("samples")

  plt.savefig(plot_path + name + '.png', dpi=150)


def plot_chroma_types(c_half, c_cqt, c_crp, chroma_labels, plot_path='./', name='chroma_types'):
  """
  plot chroma types
  """

  # pitches
  fig, ax = plt.subplots(3, 1)
  plt.subplots_adjust(hspace = 0.5)
  ax[0].imshow(c_half, aspect='auto'), ax[0].set_title("chroma halftone")
  ax[1].imshow(c_cqt, aspect='auto'), ax[1].set_title("chroma cqt")
  im = ax[2].imshow(c_crp, aspect='auto')
  ax[2].set_title("chroma crp")
  #plt.colorbar(im, ax=ax[2])

  for a in ax:
    a.set_yticks(np.arange(len(chroma_labels)))
    a.set_yticklabels(chroma_labels, fontsize=7)

  # save
  plt.savefig(plot_path + name + '.png', dpi=150)


def plot_smoothing(chroma, c_hat, c_hat_beat, chroma_labels, plot_path='./', name='smoothing'):
  """
  smoothing of chroma
  """

  fig, ax = plt.subplots(3, 1)
  plt.subplots_adjust(hspace = 0.5)
  ax[0].imshow(chroma, aspect='auto'), ax[0].set_title("chroma tuned") 
  ax[1].imshow(c_hat, aspect='auto'), ax[1].set_title("chroma without transient noise")
  ax[2].imshow(c_hat_beat, aspect='auto'), ax[2].set_title("chroma beat smoothing")

  for a in ax:
    a.set_yticks(np.arange(len(chroma_labels)))
    a.set_yticklabels(chroma_labels, fontsize=7)

  # save
  plt.savefig(plot_path + name + '.png', dpi=150)


def plot_tonal(c_hat_beat, c_proj, hcdf, chroma_labels, plot_path='./', name='tonal'):
  """
  tonal plot
  """
  # tonal plots
  fig, ax = plt.subplots(3, 1)
  plt.subplots_adjust(hspace = 0.5)
  ax[0].imshow(c_hat_beat, aspect='auto'), ax[0].set_title("chroma beat smoothed")
  ax[1].imshow(c_proj, aspect='auto'), ax[1].set_title("chroma proj")
  ax[2].imshow(hcdf[np.newaxis, :], aspect='auto'), ax[2].set_title("hcdf")

  ax[0].set_yticks(np.arange(len(chroma_labels)))
  ax[0].set_yticklabels(chroma_labels, fontsize=7)

  # save
  plt.savefig(plot_path + name + '.png', dpi=150)

def plot_est_chords(chord_hat, chord_labels, plot_path='./', name='chord_est'):
  """
  plot estimated chords
  """

  fig, ax = plt.subplots(1, 1, figsize=(9, 8))
  img = ax.imshow(chord_hat, aspect='auto')

  plt.ylabel('chord')
  plt.xlabel('beat frames')

  ax.set_yticks(np.arange(len(chord_labels)))
  ax.set_yticklabels(chord_labels, fontsize=7)

  for l in np.arange(chord_hat.shape[1]):
    plt.axvline(x=l-0.5)
  
  for l in np.arange(len(chord_hat)):
    plt.axhline(y=l-0.5)

  # save
  plt.savefig(plot_path + name + '.png', dpi=150)


if __name__ == "__main__":
  """
  main function
  """

  # plot paths
  plot_path = './ignore/ass8_data/plots/'

  # file path
  file_path = './ignore/ass8_data/01-AchGottundHerr.wav'

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

  # hop size
  hop = N//2

  # create half tone filter bank
  Hp, chroma_labels = create_half_tone_filterbank(N, fs, midi_start_note=midi_start_note, num_oct=4)

  # calculate pitches
  c_half = calc_chroma_halftone(x, Hp, N)

  # chroma with cqt with tuning
  c_cqt = calc_chroma(x, fs, hop=hop, n_octaves=5, bins_per_octave=36, fmin=libr.note_to_hz(start_note))

  # calc CRP
  c_crp = calc_crp( x, Hp, N, midi_start_note=midi_start_note)

  # choose chroma for further processing
  #chroma = c_cqt
  chroma = c_crp

  # remove transient noise
  c_hat = matrix_median(chroma, n_med=6)

  # beat snychron smoothing of chromagram 
  tempo, beats = libr.beat.beat_track(x, sr=fs, hop_length=hop)

  print("Tempo: ", tempo)
  print("beats: ", beats)
  print("beats: ", np.diff(beats))
  print("beats: ", beats.shape)

  # feature averaging over beats
  #c_hat_beat = frame_filter(c_hat, beats, filter_type='mean')
  c_hat_beat = frame_filter(c_hat, beats, filter_type='mean')


  # --
  # tonal centroid projection

  # apply projection
  c_proj = tonal_centroid_proj(c_hat_beat)

  # harmonic changes
  hcdf = harmonic_change_detection(c_proj)

  # chord mask
  M, chroma_labels, chord_labels = create_chord_mask(maj7=False, g6=False, start_note='G')

  # chord measure
  chord_hat = M @ c_hat_beat

  # chord estimation 
  chord_est = np.zeros(chord_hat.shape)

  for i, k in enumerate(np.argmax(chord_hat, axis=0)):
    chord_est[k, i] = 1.0

  # plot estimated chords
  #plot_est_chords(chord_hat, chord_labels, plot_path, name='chord_hat')
  #plot_est_chords(chord_est, chord_labels, plot_path, name='chord_est')

  # --
  # some plots

  # for key shift
  #chroma_labels = get_chroma_labels(start_note='C')

  # tonal plot
  #plot_tonal(c_hat_beat, c_proj, hcdf, chroma_labels, plot_path, name='tonal')

  # chroma plots
  plot_chroma_types(c_half, c_cqt, c_crp, chroma_labels, plot_path, name='chroma_types')
  #plot_smoothing(chroma, c_hat, c_hat_beat, chroma_labels, plot_path, name='smoothing')


  # plot Hp
  Halftone(Hp, plot_path, name='halftone_filterbank')

  # show plots
  #plt.show()

    




