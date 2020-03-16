# --
# chorus detection

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# my personal mia lib
from mia2 import *

# other stuff
from scipy.io import loadmat
from scipy import stats

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

  # extract data [samples x features]
  x = data['Xr']
  m, n = x.shape

  feature_names = [data['parameterAllName'][0, i][0] for i in range(n)]
  
  #print("x: ", x)
  #print("par: ", data['parameterAllName'][0, 0][0])
  #print("feature_names: \n", feature_names)

  #print("tables: ", tables)

  print("input data: ", x.shape)
  print("feature length: ", len(feature_names))


  # --
  # calculate pca

  x_pca = calc_pca(x)
  #plot_pca(x_pca)


  # --
  # Data Visualization

  df = pd.DataFrame(x[:100, :3], columns=feature_names[:3])

  # scatter plot
  pd.plotting.scatter_matrix(df)
  #plt.show()

  # zscore
  z = stats.zscore(x)


  # --
  # factor analysis























