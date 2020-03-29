# --
# structure analysis

import numpy as np
import matplotlib.pyplot as plt

# my personal mia lib
from mia2 import *

# other stuff
from scipy.io import loadmat


def plot_sdm(sdm, cmap='magma', plot_dir=None, emb=None, suffix=''):
  """
  plot sdm matrix (simple imshow)
  """
  plt.figure()
  plt.imshow(sdm[::-1, :], aspect='auto', cmap=cmap, extent=[0, len(sdm), 0, len(sdm)])
  plt.ylabel('frames')
  plt.xlabel('frames')
  plt.colorbar()
  plt.tight_layout()

  if plot_dir is not None:
    plt.savefig(plot_dir + 'sdm_emb-'+ str(emb) + '_' + suffix + '.png', dpi=150)

  else:
    plt.show()


def plot_tanh():
  """
  evaluate tanh function
  """
  x = np.arange(-0.25, 1.25, 0.01)

  gamma = 0.5
  lams = np.concatenate((np.linspace(1/(gamma+0.2), 1/(gamma), 3), np.linspace(1/(gamma), 1/(gamma-0.2), 3)[1:]))

  # tanh maps
  y_maps = [tanh_mapping(x, gamma=gamma, lam=lam) for lam in lams]

  plt.figure()

  # plot with different lambdas
  for y, lam in zip(y_maps, lams):
    plt.plot(x, y, label='lambda={:.2f}'.format(lam))

  plt.grid()
  plt.legend()
  plt.show()


def get_mapped_sdm(chroma, plot_dir, emb=16, read_from_file=False):
  """
  get the mapped and embedded sdm matrix
  """

  # file name
  sdm_file_name = 'sdm.npy'

  # read from file to speed things up
  if read_from_file:
    return np.load(sdm_file_name)

  # calculate sdm
  sdm_chroma = calc_sdm(chroma)
  #plot_sdm(sdm_chroma, plot_dir=plot_dir)

  emb_list = [emb]
  #emb_list = range(1, 17)

  sdm_chroma_emb = np.zeros(sdm_chroma.shape)

  # run through embedding frame list
  for emb in emb_list:

    # calc indiv embedding
    sdm_chroma_emb = calc_sdm(chroma, emb=emb)
    #plot_sdm(sdm_chroma_emb, plot_dir=plot_dir, emb=emb)

  # normalization
  sdm_chroma_emb = sdm_chroma_emb / np.max(sdm_chroma_emb)
  #print("min: {} max: {}".format(sdm_chroma_emb.min(), sdm_chroma_emb.max()) )

  # sdm mapping
  sdm_chroma_map = sdm_mapping(sdm_chroma_emb)
  #plot_sdm(sdm_chroma_map, plot_dir=plot_dir, emb=emb, suffix='tanh')

  # save file
  np.save(sdm_file_name, sdm_chroma_map)

  return sdm_chroma_map


# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/e2_data/'
  plot_dir = './ignore/e2_data/plots/'

  # file names
  file_name_data = 'seq3.mat'

  # load file
  data = loadmat(file_dir + file_name_data)

  # extract chroma features
  chroma = data['CHROM']
  print("chroma: ", chroma.shape)


  # --
  # SDM embedding and mapping

  # about tanh mapping
  #plot_tanh()

  # embedding frames
  emb = 16

  # get sdm mapped
  sdm_chroma_map = get_mapped_sdm(chroma, plot_dir, emb=emb, read_from_file=True)

  #plot_sdm(sdm_chroma_map)
  print("sdm mapped: ", sdm_chroma_map.shape)


  # --
  # Template matching

  #for w in [4, 8, 16, 3, 9, 12]:
  for w in [8]:

    # recurrence matrix
    R = calc_recurrence_matrix(sdm_chroma_map, w=w)

    # plot
    #plot_sdm(R, plot_dir=plot_dir, emb=emb, suffix='ncc-w-{}'.format(w))


  # median filter length
  n_med = 4

  # median filtering
  R_med = matrix_median(R, n_med=n_med)
  
  # plot
  #plot_sdm(R_med, plot_dir=plot_dir, emb=emb, suffix='ncc-w-{}_med-{}'.format(w, n_med))

  # otsu thresholding
  R_bin = matrix_otsu_thresh(R_med, lower_bound=0.5)

  # plot
  #plot_sdm(R_bin, cmap='Greys', plot_dir=plot_dir, emb=emb, suffix='ncc-w-{}_med-{}_bin'.format(w, n_med))
  plot_sdm(R_bin, cmap='Greys')












