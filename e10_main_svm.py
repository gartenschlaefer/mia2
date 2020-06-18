# Third party modules:---------------------------------------------------------


import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs
from scipy.io import loadmat

from mia2 import *

def fitting( X , y , kernel_type):
  
  # Fit the model, don't regularize for illustration purposes
  clf = svm.SVC( C=1.0, kernel=kernel_type, degree=3, gamma='scale', 
      random_state=0 )

  clf.fit( X, y )

  return clf

#------------------------------------------------------------------------------
def generate_circles( num_samples ):
  # make the random numbers predictable
  np.random.seed( 0 )
 
  r1 = 0.5 + 0.1 * np.random.randn( num_samples, 1 )
  r2 = 1.5 + 0.2 * np.random.randn( num_samples, 1 )
  phi = 2 * np.pi * np.random.randn( 2 * num_samples, 1 )

  x1 = np.vstack( [ r1 * np.cos( phi[ 0 : num_samples ] ) ,  
      r2 * np.cos( phi[ num_samples : ] ) ] )
  
  x2 = np.vstack( [ r1 * np.sin( phi[ 0 : num_samples ] ) ,  
      r2 * np.sin( phi[ num_samples : ] ) ] )
  
  X = np.hstack( [ x1 , x2 ] )
  y = np.ravel( np.vstack( [ ( 0 * np.ones( r1.shape )).astype( int ) , 
    ( 1 * np.ones( r2.shape )).astype( int ) ] ))

  y = list( y )

  return X , y

#------------------------------------------------------------------------------
def svm_example():
  """
  SVM example from
  https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
  """

  x = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
  print("x: ", x.shape)

  y = [0, 1, 0, 1, 0, 1]

  clf = svm.SVC(kernel='linear', C=1.0)

  clf.fit(x, y)

  print("x: ", x[0])

  print(clf.predict(x[0].reshape(1, -1)))

  w = clf.coef_[0]
  print(w)

  a = -w[0] / w[1]

  xx = np.linspace(0,12)
  yy = a * xx - clf.intercept_[0] / w[1]

  h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

  plt.scatter(x[:, 0], x[:, 1], c=y)
  plt.legend()

#------------------------------------------------------------------------------
def other_svm_example( kernel_type, plot_path, save=True ):
  """
  another svm example
  """
  num_samples = 50

  # Exercise 1: Two linear seperable classes:--------------------------------
  # - https://scikit-learn.org/stable/glossary.html#term-random-state
  name_path_1 = plot_path + '/Linear Set '
  X , y = make_blobs( n_samples=num_samples, centers=2, n_features=2, 
    random_state=42 )
  clf = fitting( X , y , kernel_type )

  # Plot the clusters
  plot_svm_contours( X , y , clf , kernel_type , name_path_1 , save )
  
  # Exercise 2: Generate two non-linear seperable sets:------------------------
  name_path_2 = plot_path + '/Circle Set '
  X , y = generate_circles( num_samples // 2 )
  clf = fitting( X , y , kernel_type )
 
  plot_svm_contours( X , y , clf , kernel_type , name_path_2 , save )

#------------------------------------------------------------------------------
def plot_svm_contours( X, y, clf, kernel, name_path, save=True ):
    """
    
    The following code is directly taken from https://bit.ly/2AqJu8o 
    
    """
    fig, ax = plt.subplots( figsize=( 8 , 6 ))
    plt.scatter( X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Set1 )

    # Plot the decision function
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_cor = np.linspace( xlim[0], xlim[1], 100 )
    y_cor = np.linspace( ylim[0], ylim[1], 100 )

    Y_cor, X_cor = np.meshgrid( y_cor, x_cor )
    grid = np.vstack( [ X_cor.ravel(), Y_cor.ravel() ] ).T
    
    z = clf.decision_function( grid ).reshape( X_cor.shape )

    # Plot decision boundary and margins
    ax.contour( X_cor, Y_cor, z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'] )

    # Plot support vectors
    ax.scatter( clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], 
        s=100, linewidth=1, facecolors='none', edgecolors='k' )

    name = 'SVM with {} kernel'.format( kernel )
    
    plt.title( name )
    if save is True:
      plt.savefig( name_path  + name + '.png', dpi=150 )
    
#------------------------------------------------------------------------------
def plot_transformed_data( x, y, clf, labels, name_path, name, kernel, save=True ):
  """ 
  plot transformed data lda data points
  """

  plt.figure( figsize=( 8, 6 ))

  #----------------------------------------------------------------------------
  # Slightly adapted code from shorturl.at/jqQUY
  
  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, x_max] [y_min, y_max].
  fig, ax = plt.subplots( figsize=( 8 , 6 ))
  
   # Plot the decision function
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  
  x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
  y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
  
  # step size in the mesh
  h = .01
  
  xx, yy = np.meshgrid( np.arange( x_min, x_max, h ), 
    np.arange( y_min, y_max, h ))
  
  Z = clf.predict( np.c_[xx.ravel(), yy.ravel()] )

  # Put the result into a color plot
  Z = Z.reshape( xx.shape )
  plt.pcolormesh( xx, yy, Z, alpha=0.1, cmap=plt.cm.Set1, antialiased=True )

  for i, c in enumerate(labels):
    plt.scatter( x[0, y==i], x[1, y==i], edgecolor='k', label=c )
  
  # Plot Layout
  plt.title('3-Class classification using Support Vector Machine with '
    '{} kernel'.format( kernel ))
  
  plt.xlabel( 'lda component 1' )
  plt.ylabel( 'lda component 2' )
  plt.axis( 'tight' )
  
  plt.legend()

  if save is True:
    plt.savefig( name_path + name + '.png', dpi=150 )

#------------------------------------------------------------------------------
if __name__ == "__main__":

  # plot paths
  plot_path = 'ignore/ass10_data/plots/'

  # load data
  data = loadmat( './ignore/ass10_data/BspDrums.mat' )

  # get data arrays x:[n x m] n samples, m features
  x  = data[ 'drumFeatures' ][0][0][0].T
  y  = data[ 'drumFeatures' ][0][0][1]

  # get shape of things
  n, m = x.shape

  # get labels
  labels = np.unique( y )

  # tranfer labels to int indices
  y = label_to_index( y , labels )

  # print some info
  print("num samples: {}, num features: {}, labels: {}".format( n, m, labels ))

  # do lda transform
  lda_transform = True
  if lda_transform:
    w, bias, x, mu_k_h, label_list = train_lda_classifier( x, y )
    print("transformed data: ", x.shape)
    
  #----------------------------------------------------------------------------
  # Flag for saving all the figures
  save_fig = True

  # Different kernel types
  kernel = [ 'linear', 'poly', 'rbf', 'sigmoid' ]

  for elem in kernel:
    other_svm_example( elem , plot_path , save=save_fig )

    # Performs fitting of the transformed drum data set
    clf = svm.SVC( C=1.0, kernel=elem, degree=3, gamma='scale', 
      random_state=0 )
    clf.fit( x, y )   

    plot_transformed_data( x, y, clf, labels, plot_path, 'lda_'+elem, 
      elem, save=save_fig )
 