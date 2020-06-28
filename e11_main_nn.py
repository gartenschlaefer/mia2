# Third party modules:---------------------------------------------------------
import torch 
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.io import loadmat
from mia2 import label_to_index

if __name__ == "__main__":
    
    # Part 1 - Loading---------------------------------------------------------
    
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

    X_tensor = torch.from_numpy( X )
    y_tensor = torch.from_numpy( y )