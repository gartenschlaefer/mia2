# Torch Module:----------------------------------------------------------------
import torch 
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler

# Import User defined MLP and customData class:--------------------------------
import MLP
import customData

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
    file_path = './ignore/ass11_data/BspDrums.mat'
    
    data = loadmat( file_path )
    data_set = customData.CustomDataSetFromMat( file_path )

    # Part 2 - Set default torch model data type:------------------------------
    torch.set_default_dtype( torch.float64 )

    # Part 3 - Instantiate Neural Net:-----------------------------------------
    in_dim, hid_dim, out_dim = ( 45, 2, 3 )
    net = MLP.MLP_Net( in_dim, hid_dim, out_dim )

    # Print all net parameters onto the screen
    # print( "Neural Network parameters {}".format( list( net.parameters( ) ) ) )

    # Define a loss function and choose an optimizer
    criterion = torch.nn.MSELoss( )
    optimizer = optim.SGD( net.parameters( ), lr=0.001, momentum=0.9 )

    # Define number of epochs
    num_epochs = 10

    # Part 4 - Generate Training and Test set:---------------------------------
    # The following code is based on https://bit.ly/3dAxv5S    
    train, valid, test = data_set.generate_sets( data_set.__len__( ), 
        0.7, 0.15 ) 

    # For more information
    # https://pytorch.org/docs/stable/data.html#dataset-types
    train_sampler = SubsetRandomSampler( train )
    valid_sampler = SubsetRandomSampler( valid )
    test_sampler = SubsetRandomSampler( test )
