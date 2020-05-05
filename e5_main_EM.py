# Third party modules:---------------------------------------------------------
import numpy as np

from scipy.io import loadmat
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

# sklearn imports for comparison
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def compute_posterior( alpha, num_samples, num_components, num_centers, 
    X, Mu, Sigma ):
    
    prob_xn_theta_k = np.empty(( num_components, num_samples ))

    # computing the probabilities for each component
    for k in range( num_centers ):
        for n in range( num_samples ):
            distribution = multivariate_normal( Mu[ :, k ], Sigma[ :, :, k ] )
            prob_xn_theta_k[ k , n ] = distribution.pdf( X[ :, n ] ) 
    
    # r is a 2 x 200 matrix. Each entry is the posterior probability of 
    # cluster k given xn = X[ : , n ] and mu_k = Mu[ : , k ] and 
    # Sigma_k = Sigma[ : , : , k ]
    R = np.repeat( alpha , num_samples , axis=1 ) * prob_xn_theta_k
    R = R / np.sum( R , axis=0 )

    R[ np.isnan(R) ] = 1e-3
    R[ R < 1e-3 ] = 1e-3

    return R

def em_algorithm( X, num_centers, max_iter ):
    
    #Initialization:-----------------------------------------------------------
    # X is the data matrix with two components ( x , y ) and 200 samples
    # X[ 0 , : ] -> x coordinates for each data sample
    # X[ 1 , : ] -> y coordinates for each data sample 
    num_components , num_samples = X.shape

    # alpha are the weights for each cluster and is a 2 x 1 matrix
    # alpha = np.random.uniform( low=0, high=1, size=( num_centers , 1 ))
    alpha = np.array( [0.5 , 0.5] ).reshape( ( num_centers , 1 ))

    # Mu is the matrix containing all the means for each cluster
    # Mu[ 0 , : ] -> components ( x , y) for cluster 1 
    # Mu[ 1 , : ] -> components ( x , y) for cluster 2
    # Mu has the dimensions: dim( Mu ) = num_components x num_centers
    Mu = np.random.uniform( low=-1, high=1, size=( num_components, 
        num_centers ))

    # Sigma is the covariance matrix of dimension 2 x 2 x 2
    # Sigma[ : , : , 0 ] -> 2 x 2 covariance matrix of cluster 1 with Mu 1
    # Sigma[ : , : , 1 ] -> 2 x 2 covariance matrix of cluster 2 with Mu 2
    init_Sigma = np.random.uniform( low=-1, high=1, size=( num_components , 
        num_centers))

    # Make the main diagonal more prominent
    Sigma = np.cov( init_Sigma ) + 2*np.eye( num_centers )
    Sigma = np.repeat( Sigma[ :, :, np.newaxis ], num_centers , axis=2 )

    # E-Step: Computing the posterior probabilities:---------------------------
    counter = 0
    while counter <= max_iter:
        R = compute_posterior( alpha, num_samples, num_components, 
            num_centers, X, Mu, Sigma )

        # M-Step: Updating the parameters:---------------------------------
        Nk = np.einsum( 'ij->i', R )

        dimension = ( num_components, num_centers, num_centers )
        Nk_ext = np.repeat(Nk,num_components*num_centers).reshape(dimension)
    
        alpha = Nk.reshape(( num_centers, 1 )) / num_samples
        Mu = np.einsum( 'ij,nj->in', R, X ) / Nk

        test_1 = np.einsum( 'ij,ik->ijk' , X.T, X.T )
        test_2 = np.einsum( 'ijl,ik->kjl', test_1, R.T ) / Nk_ext
        test_3 = np.einsum( 'mn,ml->mnl' , Mu, Mu )
        Sigma = ( test_2 - test_3 )
        
        counter += 1

    return Mu, Sigma

def visualization( X ):

    fig, ax = plt.subplots()
    
    ax.scatter( X[ 0 , : ] , X[ 1 , : ], s=10 ,alpha=0.5 )

    ax.set_xlabel( r'$x_1$', fontsize=16 )
    ax.set_ylabel( r'$x_2$', fontsize=16 )
    ax.set_title( 'Test Data Set' )

    ax.grid( True )
    fig.tight_layout()

    plt.show()

if __name__ == "__main__":

    annotations = loadmat( 'EM_data.mat' )
    
    # Covariance matrices and Mean vectors for cluster 1 and 2
    # For comparisions! Not needed for the EM-Algorithm
    S_1 = annotations[ 'S1' ]
    S_2 = annotations[ 'S2' ]
    mu_1 = annotations[ 'mu1' ]
    mu_2 = annotations[ 'mu2' ]

    # The data itself
    X = annotations[ 'X' ]

    # Make a scatter plot for visualization
    # visualization( X )

    # EM-Algorithm
    num_centers = 2
    Mu, Sigma = em_algorithm( X, num_centers, max_iter=10 )