# Third party modules:---------------------------------------------------------


import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
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




  # --
  # some plots or other stuff

  # test svm
  #svm_example()


  # show plots
  plt.show()