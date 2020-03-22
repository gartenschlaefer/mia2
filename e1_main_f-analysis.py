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

# Factor analysis module written by Jeremy Biggs
# For further information, see:
#
# https://pypi.org/project/factor-analyzer/
# https://factor-analyzer.readthedocs.io/en/latest/index.html
# https://factor-analyzer.readthedocs.io/en/latest/factor_analyzer.html
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
def plot_scree( eigen_values ):
  plt.scatter( range( 1, eigen_values.shape[0] + 1) , eigen_values )
  plt.plot( range( 1, eigen_values.shape[0] + 1) , eigen_values )
  plt.title( "Scree Plot" )
  plt.xlabel( "Number of eigenvalues" )
  plt.ylabel( "Eigenvalue" )
  plt.grid()
  plt.show()

#------------------------------------------------------------------------------
def plot_scatter_matrix(x, M=100, N=3):
  """
  plot a scatter matrix with pandas with M samples and N features
  """
  pd.plotting.scatter_matrix( pd.DataFrame(x[:M, :N], columns=feature_names[:N] ))
  plt.show()

#------------------------------------------------------------------------------
def plot_dendrogram(z):
  """
  plot a dendrogram
  """
  plt.figure()
  dn = hierarchy.dendrogram( hierarchy.linkage(z, 'single') )
  plt.show()

#------------------------------------------------------------------------------
def plot_corr(c):
  """
  make image plot of correlation
  """
  plt.figure()
  plt.imshow(c)
  plt.colorbar()
  plt.tight_layout()
  plt.show()

#------------------------------------------------------------------------------
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
  print("input data: ", x.shape)
  print("feature length: ", len(feature_names))

  #------------------------------------------------------------------------------
  # Calculate pca and obtain eigen-values, cross-correlation vector and 
  # covariance matrix.
  x_pca, eigen_values = calc_pca( x )
  z = stats.zscore( x )
  r = np.corrcoef( z.T )
  C = np.cov( z.T )
  
  #------------------------------------------------------------------------------
  # Data Visualization

  plot_scatter_matrix( x )  # Plot scatter matrix
  plot_dendrogram( z.T )    # Plot dendrogram
  plot_corr( r )            # Plot correlation
  plot_corr( C )            # PLot covariance

  # Plot the scatter-plot for the components
  plot_pca( x_pca )

  # Plot the scree-plot for the componets
  # Note:
  #   - Eigenvalues are not normalized!
  #   - Corresponding values are high!
  plot_scree( eigen_values )

  #------------------------------------------------------------------------------
  # Adequacy test, needed to evaluate the 
  # factorabiltiy of the given data set.

  # Option One - Barlett's Test: 
  chi_square_value, p_value = calculate_bartlett_sphericity( x )
  print("chi-square value: {}\np-value: {}\n".format( 
    chi_square_value, p_value ))
  
  # Option Two - Kaiser-Meyer-Olkin (KMO) Test:
  kmo_all, kmo_model = calculate_kmo( x )
  print("kmo-model score: {}".format( kmo_model ))

  #------------------------------------------------------------------------------
  # Performing factor analysis
  # By inspecting the scree plot, one can see, that only the
  # first two eigenvalues have the biggest variance. 
  # 
  # Therefore the other 26 can be neglected for the analysis
  fa = FactorAnalyzer( n_factors=2, rotation="varimax", method="ml" )
  fa.fit( x )
  
  correlation_matrix = fa.corr_
  factor_correlation_matrix = fa.phi_
  factor_loading_matrix = fa.loadings_
  rotation_matrix = fa.rotation_matrix_
  