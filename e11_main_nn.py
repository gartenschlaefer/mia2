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
    data_set = customData.CustomDataSetFromMat( file_path, 'drumFeatures' )
    
    # Convert drum_labels to integers
    unique_flags = np.unique( data_set.drum_labels )
    
    data_set.drum_labels = label_to_index( data_set.drum_labels, unique_flags )
    data_set.drum_labels = torch.tensor( data_set.drum_labels, 
        dtype=torch.float64, requires_grad=False )
      
    # Part 2 - Set default torch model data type:------------------------------
    torch.set_default_dtype( torch.float64 )

    # Part 3 - Instantiate Neural Net:-----------------------------------------
    in_dim, hid_dim, out_dim = ( 45, 2, 3 )
    net = MLP.MLP_Net( in_dim, hid_dim, out_dim )

    # Print all net parameters onto the screen
    # print( "Neural Network parameters {}".format( list( net.parameters( ) ) ) )

    # Define a loss function and choose an optimizer
    #criterion = torch.nn.MSELoss( reduction='mean' )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD( net.parameters( ), lr=1e-5, momentum=0.8 )

    # Part 4 - Generate Training and Test set:---------------------------------
    # The following code is based on https://bit.ly/3dAxv5S    
    train, valid, test = data_set.generate_sets( data_set.__len__( ), 
        0.6, 0.2 ) 

    #print("train: ", train.shape)

    # For more information:
    # - https://bit.ly/2NzOASO
    # 
    # For SubsetRandomSampler(  ):
    # - https://bit.ly/3eL1cm3 
    train_sampler = SubsetRandomSampler( train )
    valid_sampler = SubsetRandomSampler( valid )
    test_sampler = SubsetRandomSampler( test )

    batch_size = 2
    train_loader = torch.utils.data.DataLoader( data_set, batch_size=batch_size,
        num_workers=0, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader( data_set, batch_size=batch_size,
        num_workers=0, sampler=valid_sampler)

    testloader = torch.utils.data.DataLoader( data_set, batch_size=batch_size,
        num_workers=0, sampler=test_sampler )

    # Train the network
    num_epochs = 100
    for epoch in range( num_epochs ):
        running_loss = 0.0
        for i , data in enumerate( train_loader, 0 ):
            # get the inputs; data is a list of [ inputs, labels ]
            inputs, labels = data

            # conversion of data type
            inputs = torch.tensor(inputs, dtype=torch.double)
            labels = torch.tensor(labels, dtype=torch.long)

            # Zero the parameter gradients, otherwise we would 
            # accumulate the gradients for each loop iteration! 
            optimizer.zero_grad(  )

            # Forward + Backward + optimize
            outputs = net( inputs )

            # loss
            loss = criterion( outputs, labels )

            #print("inputs: ", inputs), print("outputs: ", outputs), print("labels: ", labels), print("loss: ", loss)

            loss.backward(  )
            optimizer.step(  )

            # print statistics
            running_loss += loss.item(  )
            if i % 100 == 99:
                print( "[%d, %5d] loss %.3f" % ( epoch + 1 , i + 1, 
                    running_loss / 10 ) )
                running_loss = 0.0

    print( 'Finished Training' )

    # --
    # evaluate whole dataset

    # metric init
    correct = 0
    total = 0

    # no gradients for eval
    with torch.no_grad():

        # load data
        for data in testloader:

            # extract sample
            inputs, labels = data

            # classify
            outputs = net(inputs)

            # prediction
            _, predicted = torch.max(outputs.data, 1)

            #print("predicted: ", predicted[0].item())
            #print("l: ", labels[0].item())

            # add total amount of prediction
            total += labels.size(0)

            # check if correctly predicted
            correct += (predicted == labels).sum().item()

    # plot accuracy
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
                  
