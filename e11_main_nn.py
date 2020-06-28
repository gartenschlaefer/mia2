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
    
    n , m = X.shape  
    
    # get labels
    labels = np.unique( y )
    y = label_to_index( y, labels )
    
    # print some info
    print("num samples: {}, num features: {}, labels: {}".
        format( n, m, labels ) )

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
    print( "Neural Network parameters {}".format( list( net.parameters( ) ) ) )

    # Define a loss function and choose an optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD( net.parameters(), lr=0.001, momentum=0.9 )