# Third party modules:---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs

def plot_svm_contours( X, clf, kernels ):
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

if __name__ == "__main__":
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
    plot_svm_contours( X , clf , kernels)
    

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
