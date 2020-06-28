# Torch Module:----------------------------------------------------------------
import torch 
import torch.optim as optim

# Import User defined MLP class:-----------------------------------------------
import MLP

# Import User defined label_to_index function:---------------------------------
from mia2 import label_to_index

# Import numpy, matplotlib and scipy helper functions:-------------------------
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.io import loadmat

#------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Part 1 - Loading:--------------------------------------------------------
    plot_path = 'ignore/ass10_data/plots/'
    data = loadmat( './ignore/ass11_data/BspDrums.mat' )

    X  = data[ 'drumFeatures' ][0][0][0]
    y  = data[ 'drumFeatures' ][0][0][1] 
    
    # N = total number of samples
    # M = total number of features
    N , M = X.shape  
    
    # get labels
    labels = np.unique( y )
    y = label_to_index( y, labels )
    
    # print some info
    print( "num samples: {}, num features: {}, labels: {}".
        format( N, M, labels ) )

    # Part 2 - Convert to Torch tensors:---------------------------------------
    torch.set_default_dtype( torch.float64 )

    # Possibility 1
    # X = torch.as_tensor( X, dtype=torch.float64 ).requires_grad_(  )
    # y = torch.as_tensor( y, dtype=torch.float64 )

    # Possibility 2
    X = torch.tensor( X, dtype=torch.float64, requires_grad=True ) 
    y = torch.tensor( y, dtype=torch.int64, requires_grad=False )

    # Part 3 - Instantiate Neural Net:-----------------------------------------
    in_dim = 45
    hid_dim = 2
    out_dim = 3
    net = MLP.MLP_Net( in_dim, hid_dim, out_dim )

    # Print all net parameters onto the screen
    # print( "Neural Network parameters {}".format( list( net.parameters( ) ) ) )

    # Define a loss function and choose an optimizer
    criterion = torch.nn.MSELoss( )
    optimizer = optim.SGD( net.parameters( ), lr=0.001, momentum=0.9 )

    # Define number of epochs
    num_epochs = 10

    # Part 4 - Generate Training and Test set:---------------------------------
    train_ratio = int( 0.7 * N  )
    valid_ratio = int( 0.15 * N )
    test_ratio  = int( 0.15 * N ) 

    num_samples = train_ratio + valid_ratio + test_ratio
    diff = N - num_samples

    # Here, we make sure that the corresponding ratios 
    # sum up to the total number of data samples, because 
    # torch.utils.data.random_split( dataset, lengths) lengths
    # parameter needs a sequence e.g a Python List of integers
    if diff < 0 :   train_ratio -= diff
    elif diff > 0 : train_ratio += diff
    ratios = [ train_ratio, valid_ratio, test_ratio ]
    
    train_set, valid_set, test_set = torch.utils.data.random_split(
       X, ratios )

    print( len( train_set ) )
    print( len( valid_set ) )
    print( len( test_set ) )
