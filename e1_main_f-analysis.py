# --
# factor-analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# my personal mia lib
from mia2 import *

# other stuff
from scipy.io import loadmat
from scipy import stats
from scipy.cluster import hierarchy

# 3d plot
from mpl_toolkits.mplot3d import Axes3D

def plot_pca(x_pca):
  """
  plot pca in 2d and 3d
  """
  # 2d
  plt.figure(1)
  plt.scatter(x_pca[:, 0], x_pca[:, 1])
  plt.grid()
  plt.xlabel("PCA component 1")
  plt.ylabel("PCA component 2")
  plt.tight_layout()

  # 3d
  fig = plt.figure(2)
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2])
  ax.set_xlabel("PCA component 1")
  ax.set_ylabel("PCA component 2")
  ax.set_zlabel("PCA component 3")
  plt.tight_layout()
  plt.show()

def plot_scatter_matrix(x, M=100, N=3):
  """
  plot a scatter matrix with pandas with M samples and N features
  """
  pd.plotting.scatter_matrix( pd.DataFrame(x[:M, :N], columns=feature_names[:N] ))
  plt.show()

def plot_dendrogram(z):
  """
  plot a dendrogram
  """
  plt.figure()
  dn = hierarchy.dendrogram( hierarchy.linkage(z, 'single') )
  plt.show()

def plot_corr(c):
  """
  make image plot of correlation
  """
  plt.figure()
  plt.imshow(c)
  plt.colorbar()
  plt.tight_layout()
  plt.show()

# --
# Main function
if __name__ == '__main__':

  # read audiofile
  file_dir = './ignore/e1_data/Cars/'

  # file names
  file_name_data = 'data.mat'
  file_name_tables = 'tables.mat'

  # load file
  data = loadmat(file_dir + file_name_data)
  #tables = loadmat(file_dir + file_name_tables)
  #print("tables: ", tables)

  # extract data [samples x features]
  x = data['Xr']
  m, n = x.shape

  feature_names = [data['parameterAllName'][0, i][0] for i in range(n)]
  
  #print("feature_names: \n", feature_names)

  print("input data: ", x.shape)
  print("feature length: ", len(feature_names))


  # --
  # calculate pca

  x_pca = calc_pca( x )
  #plot_pca(x_pca)

  # --
  # Data Visualization

  # zscore
  z = stats.zscore(x)

  # scatter matrix
  #plot_scatter_matrix(x)

  # dendrogram
  #plot_dendrogram(z.T)

  # correlation
  r = np.corrcoef(z.T)
  plot_corr(r)

  # covariance
  C = np.cov(z.T)
  plot_corr(C)

  # --
  # factor analysis

