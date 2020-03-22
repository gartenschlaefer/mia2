# --
# structure analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# my personal mia lib
from mia2 import *

# other stuff
from scipy.io import loadmat
from scipy import stats
from scipy.cluster import hierarchy


def plot_sdm(sdm, vmin=None, vmax=None, cmap='magma', plot_dir=None, emb=None):
  """
  plot sdm matrix (simple imshow)
  """
  plt.figure()
  plt.imshow(sdm[::-1, :], aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap, extent=[0, len(sdm), 0, len(sdm)])
  plt.ylabel('frames')
  plt.xlabel('frames')
  plt.colorbar()
  plt.tight_layout()

  if plot_dir is not None:
    plt.savefig(plot_dir + 'sdm_emb-'+ str(emb) + '.png', dpi=150)

  else:
    plt.show()


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


  # --
  # extract features

  # chroma features
  chroma = data['CHROM']
  print("chroma: ", chroma.shape)


  # --
  # SDM normal

  sdm_chroma = calc_sdm(chroma)
  print("sdm: ", sdm_chroma.shape)
  #plot_sdm(sdm_chroma, plot_dir=plot_dir)


  # --
  # Embedding Dimensions

  emb_list = [16]
  #emb_list = range(1, 17)

  sdm_chroma_emb = np.zeros(sdm_chroma.shape)

  for emb in emb_list:
    sdm_chroma_emb = calc_sdm(chroma, emb=emb)
    print("sdm emb: ", sdm_chroma_emb.shape)
    #plot_sdm(sdm_chroma_emb, plot_dir=plot_dir, emb=emb)


  # --
  # Reduction of irrelevant things

  # normalization
  sdm_chroma_emb = sdm_chroma_emb / np.max(sdm_chroma_emb)
  print("min: {} max: {}".format(sdm_chroma_emb.min(), sdm_chroma_emb.max()) )

  # sdm mapping
  sdm_chroma_map = sdm_mapping(sdm_chroma_emb)
  plot_sdm(sdm_chroma_emb)
























