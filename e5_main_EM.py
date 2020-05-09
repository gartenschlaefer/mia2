# Third party modules:---------------------------------------------------------
import numpy as np

from scipy.io import loadmat
from scipy.stats import multivariate_normal

import matplotlib 
import matplotlib.pyplot as plt

# sklearn imports for comparison
from sklearn.cluster import KMeans

#------------------------------------------------------------------------------
def compute_posterior( alpha, num_samples, num_components, num_centers, 
    X, Mu, Sigma ):
    
    prob_xn_theta_k = np.empty(( num_components, num_samples ))

    # computing the probabilities for each component
    for k in range( num_centers ):
        #print info
        print("\n--k: ", k)
        print("mu k: \n", Mu[k, :])
        print("Sigma k: \n", Sigma[k, :, :])

        # set distribution
        distribution = multivariate_normal( Mu[k, :], Sigma[ k, :, : ] )

        for n in range( num_samples ):
            
            # calculate probability
            prob_xn_theta_k[ k , n ] = distribution.pdf( X[ :, n ] ) 
    
    # r is a 2 x 200 matrix. Each entry is the posterior probability of 
    # cluster k given xn = X[ : , n ] and mu_k = Mu[ : , k ] and 
    # Sigma_k = Sigma[ k , : , : ]
    R = np.repeat( alpha , num_samples , axis=1 ) * prob_xn_theta_k
    R = R / np.sum( R , axis=0 )

    # Line 41 and 42 make sure, that there are no invalid values 
    # in th resulting matrix 
    R[ np.isnan(R) ] = 1e-3
    R[ R < 1e-3 ] = 1e-3

    return R

#------------------------------------------------------------------------------
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
    # Sigma[ 0, : , : ] -> 2 x 2 covariance matrix of cluster 1 with Mu 1
    # Sigma[ 1, : , : ] -> 2 x 2 covariance matrix of cluster 2 with Mu 2
    init_Sigma = np.random.uniform( low=-1, high=1, size=( num_components , 
        num_centers))

    # Make the main diagonal more prominent
    Sigma = init_Sigma + 2*np.eye( num_centers )

    # [k x m x m]
    Sigma = np.repeat( Sigma[np.newaxis, :, :], num_centers , axis=0 )

    # E-Step: Computing the posterior probabilities:---------------------------
    counter = 0
    while counter <= max_iter:
        print("\n---\niteration: ", counter)

        # E-Step: Expectation [k x n]
        R = compute_posterior( alpha, num_samples, num_components, 
            num_centers, X, Mu, Sigma )

        # M-Step: Updating the parameters:---------------------------------
        
        # summed R over n [k]
        Nk = np.einsum( 'ij->i', R )

        # weights alpha for each kernel k [k, 1]
        alpha = Nk.reshape(( num_centers, 1 )) / num_samples

        # --
        # some dimensions:
        # --
        #   R: [k x n]  k kernels
        #   X: [m x n]  m features, n samples
        #   Mu:[k x m]
        Mu = np.einsum( 'kn, mn -> km', R, X ) / Nk

        # [k x m x n] = [m x n] - [k x m x new]
        x_mu = X - Mu[:, :, np.newaxis]

        # covar [k x m x m]
        Sigma = ( np.einsum('kn, kmn, kqn -> kmq', R, x_mu, x_mu) 
            / Nk[:, np.newaxis, np.newaxis] )
        counter += 1

    return Mu, Sigma

#------------------------------------------------------------------------------
def visualization_gmm( X, Mu, Sigma, kernels, num_centers, max_iter ):
    """ The visualization_gmm is adepted from the 
    following code example respectively sources:

    - https://bit.ly/3bc7Uil 
    Gitlab respository were I worked on as well - Nico Seddiki
    
    - https://bit.ly/2WBlmHf:
    Official documentation for multivariate_normal function.

    """

    # Initialize plots:--------------------------------------------------------
    fig, ax = plt.subplots()
    ax.scatter( X[ 0 , : ] , X[ 1 , : ] , s=10 , alpha=0.5 )

    # number of points respectively data samples
    nr_points = np.size( X[ 0, : ] )

    # Boundaries for plot
    ( xmin , xmax ) = ( -3 , 3 ) 
    ( ymin , ymax ) = ( -3 , 3 )

    delta_x = float( xmax - xmin ) / float( nr_points )
    delta_y = float( ymax - ymin ) / float( nr_points )
    x = np.arange( xmin, xmax, delta_x)
    y = np.arange( ymin, ymax, delta_y)
    
    x1, x2 = np.meshgrid( x, y )

    # Get reference mu for each cluster and plot it:---------------------------
    for key in kernels.keys():
        for center in range( num_centers ):
            coordinates_mu = kernels[key]

            ax.scatter( coordinates_mu[0], coordinates_mu[1], s=30,
                alpha=0.5, color='red', marker='x', antialiased=True )

            pos =  np.dstack( (x1, x2) )
            x3  =  multivariate_normal.pdf( pos , 
                Mu[center, :], Sigma[ center, :, : ] )

            contour  = plt.contour( x1, x2, x3 )

    ax.set_xlabel( r'$x_1$', fontsize=16 )
    ax.set_ylabel( r'$x_2$', fontsize=16 )
    ax.clabel( contour, inline=1, fontsize=10)
    ax.set_title( 'EM-Clustering Technique: {} Iterations'.format( max_iter ))

    ax.grid( True )
    fig.tight_layout()

    plt.show()

#------------------------------------------------------------------------------
def visualization_kmeans( X_kmeans, labels, cluster_centers, max_iter ):

    fig, ax = plt.subplots()
    
    ax.scatter( X_kmeans[:,0] , X_kmeans[:,1] , s=10 , alpha=0.5 , c=labels, 
        cmap='viridis' )
    ax.scatter( cluster_centers[: , 0], cluster_centers[: , 1], s=30,
        alpha=0.5, color='red', marker='x', antialiased=True )
    
    ax.set_xlabel( r'$x_1$', fontsize=16 )
    ax.set_ylabel( r'$x_2$', fontsize=16 )
    ax.set_title('KMeans-Clustering Technique: {} Iterations'.format(max_iter))

    ax.grid( True )
    fig.tight_layout()

    plt.show()

#------------------------------------------------------------------------------
if __name__ == "__main__":
    annotations = loadmat( './ignore/ass5_data/EM_data.mat' )
    
    # EM-Algorithm:------------------------------------------------------------
    # Covariance matrices and Mean vectors for cluster 1 and 2
    # For comparisions! Not needed for the EM-Algorithm
    S_1  = annotations[ 'S1' ]
    S_2  = annotations[ 'S2' ]
    mu_1 = annotations[ 'mu1' ]
    mu_2 = annotations[ 'mu2' ]

    # Dictionary containing the cluster centers
    kernels = { 'K1' : mu_1  , 'K2' : mu_2  }

    # The data itself
    X = annotations[ 'X' ]

    # EM-Algorithm
    max_iter = [ 1, 5, 10, 25, 50, 75, 100 ]
    num_centers = 2

    # Full EM-Algorithm and plot
    # for i in max_iter:
    #     Mu, Sigma = em_algorithm( X, num_centers, i )
    #     visualization_gmm( X, Mu, Sigma, kernels, num_centers, i )
   
    # KMeans-Algorithm:--------------------------------------------------------
    # The following is adapted and taken from the official sklearn 
    # documentation:
    # - https://bit.ly/35MREmP ,
    # - https://bit.ly/35JYwl9 ,

    X_kmeans = X.T
    kmeans = KMeans( n_clusters=num_centers, init='k-means++', 
        max_iter=max_iter[-1] )
    kmeans.fit( X_kmeans )
    
    visualization_kmeans( X_kmeans, kmeans.labels_, kmeans.cluster_centers_,
        kmeans.n_iter_ )