# Third party modules:---------------------------------------------------------
import numpy as np
import torch as pyt

import matplotlib
import matplotlib.pyplot as plt

from scipy.io import loadmat
from mia2 import label_to_index

if __name__ == "__main__":
    
    # plot paths
    plot_path = 'ignore/ass10_data/plots/'

    # load data
    data = loadmat( './ignore/ass11_data/BspDrums.mat' )

    # get data arrays x:[n x m] n samples, m features
    X  = data[ 'drumFeatures' ][0][0][0].T
    
    # get the data labels
    y  = data[ 'drumFeatures' ][0][0][1]
    
    # get shape of things
    n, m = X.shape  
    
    # get labels
    labels = np.unique( y ) 
    
    # tranfer labels to int indices
    y = label_to_index( y, labels )    
    
    # print some info
    print("num samples: {}, num features: {}, labels: {}".
        format( n, m, labels ) )
    