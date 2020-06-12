# Third party modules:---------------------------------------------------------


import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs
from scipy.io import loadmat

from mia2 import *


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


def other_svm_example():
  """
  another svm example
  """
  num_samples = 50

  # Exercise 1: Two linear seperable classes:--------------------------------
  kernels = [ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]

  # Generate a linear seperable data set
  # X ... The generated samples.
  # y ... The integer labels for cluster membership of each sample.

  # Try different random state values as well!
  # - https://scikit-learn.org/stable/glossary.html#term-random-state
  X, y = make_blobs( n_samples=num_samples, centers=2, n_features=2 )

  # Fit the model, don't regularize for illustration purposes
  clf = svm.SVC( C=1.0, kernel=kernels[0], degree=3, gamma='scale', 
      random_state=0 )
  clf.fit( X, y )

  # Plot the clusters
  plot_svm_contours( X, y , clf , kernels)
  

  # Exercise 2: Two non-linear seperable classes:----------------------------

  # Option 1 - Per Hand (see provided Matlabfile )
  # Parameters for the synthetic data
  r1 = 0.5 + 0.1 * np.random.randn( num_samples, 1 )
  r2 = 1.5 + 0.2 * np.random.randn( num_samples, 1 )
  phi = 2 * np.pi * np.random.randn( 2 * num_samples, 1 )

  x1 = np.vstack( [ r1 * np.cos( phi[ 0 : num_samples ] ) ,  
      r2 * np.cos( phi[ num_samples : ] ) ] )
  
  x2 = np.vstack( [ r1 * np.sin( phi[ 0 : num_samples ] ) ,  
      r2 * np.sin( phi[ num_samples : ] ) ] )
  
  data = np.hstack( [ x1 , x2 ] )

  plt.scatter( data[ :, 0 ], data[ :, 1 ] )
  plt.show()

  # 
  # % Klassenlabels
  # % Y=[num2str(ones(N,1));num2str(2*ones(N,1));num2str(3*ones(N,1))];
  # Y = num2str(reshape((ones(N,1).*[1 2]),[],1));
  # % in Y stehen jeweils pro Datensatz der dazugehörige Label ("1", "2") 
  # idX = reshape(1:length(Y),30,2);
  # id1 = idX(:,1);
  # id2 = idX(:,2);


def plot_svm_contours( X, y, clf, kernels ):
    """
    
    The following code is directly taken from https://bit.ly/2AqJu8o 
    
    """
    
    plt.scatter( X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Set1 )

    # Plot the decision function
    ax = plt.gca()
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

    plt.title( 'SVM with {} kernel'.format( kernels[0] ))
    plt.show()


def plot_transformed_data(x, y, labels, plot_path, name):
  """
  plot transformed data lda data points
  """

  plt.figure( figsize=(8, 6) )

  for i, c in enumerate(labels):
   plt.scatter( x[0, y==i], x[1, y==i], edgecolor='k', label=c)

  #plt.scatter( x[0], x[1], c=y, cmap=plt.cm.Blues, edgecolor='k', label=labels)

  plt.xlabel( 'lda component 1' )
  plt.ylabel( 'lda component 2' )
  plt.legend()
  plt.grid()

  plt.savefig( plot_path + name + '.png', dpi=150 )



if __name__ == "__main__":

  # plot paths
  plot_path = 'ignore/ass10_data/plots/'

  # load data
  data = loadmat( './ignore/ass10_data/BspDrums.mat' )

  # get data arrays x:[n x m] n samples, m features
  x  = data[ 'drumFeatures' ][0][0][0]
  y  = data[ 'drumFeatures' ][0][0][1]

  # get shape of things
  n, m = x.shape

  # get labels
  labels = np.unique(y)

  # tranfer labels to int indices
  y = label_to_index(y, labels)

  # print some info
  print("num samples: {}, num features: {}, labels: {}".format(n, m, labels))


  # lda
  lda_transform = True

  # do lda transform
  if lda_transform:

    w, bias, x, mu_k_h, label_list = train_lda_classifier(x, y)

    print("transformed data: ", x.shape)

    # plot transformed data x_h = [k-1, n]
    #plot_transformed_data(x, y, labels, plot_path, 'lda')


  other_svm_example()



  # --
  # some plots or other stuff

  # test svm
  #svm_example()


  # show plots
  plt.show()



