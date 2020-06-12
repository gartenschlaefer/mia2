# Third party modules:---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs

if __name__ == "__main__":

    # Exercise 1, Two linear seperable classes:--------------------------------
    kernels = [ 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ]

    # Generate a linear seperable data set
    # X ... The generated samples.
    # y ... The integer labels for cluster membership of each sample.

    # Try different random state values as well!
    # - https://scikit-learn.org/stable/glossary.html#term-random-state
    X, y = make_blobs( n_samples=100, centers=2, n_features=2 )

    # Fit the model, don't regularize for illustration purposes
    clf = svm.SVC( C=1.0, kernel=kernels[0], degree=3, gamma='scale', 
        random_state=0 )
    clf.fit( X, y )
    
    #--------------------------------------------------------------------------
    # The following code is directly taken from https://bit.ly/2AqJu8o 
    # Plot the clusters
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

    plt.show()