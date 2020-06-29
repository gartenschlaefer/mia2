import numpy as np

def generate( dataset_size ):    

    indices = list( range( dataset_size ) )
    
    random_seed = 42
    np.random.seed( random_seed )

    # Splits for the different sets
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = int( ( 1 - train_ratio ) // valid_ratio )

    train_split = int( np.ceil( train_ratio * dataset_size ) )
    test_split =  int( ( dataset_size - train_split ) // test_ratio )
    test_split += train_split
    
    # randomly shuffle the indices
    np.random.shuffle( indices )

    train_indices = indices[ : train_split ]
    valid_indices = indices[ train_split : test_split ]
    test_indices = indices[ test_split : ] 

    return train_indices, valid_indices, test_indices