import numpy as np

from scipy.io import loadmat
from torch.utils.data import Dataset

class CustomDataSetFromMat( Dataset ):
    def __init__( self, mat_path, key ):
        
        super( CustomDataSetFromMat, self ).__init__(  )
        
        self.data = loadmat( mat_path )
        self.drum_feats = self.data[ key ][0][0][0]
        self.drum_labels = self.data[ key ][0][0][1] 
    
    def __getitem__( self, index ):
        drum_sound = self.drum_feats[ index, : ]
        label = self.drum_labels[ index ]

        return drum_sound, label

    def __len__( self ):
        return len( self.drum_labels )

    @staticmethod
    def generate_sets( dataset_size, train_ratio, valid_ratio ):
    
        random_seed = 42
        np.random.seed( random_seed )
         
        indices = list( range( dataset_size ) )

        test_ratio = int( ( 1 - train_ratio ) // valid_ratio )
        train_split = int( np.ceil( train_ratio * dataset_size ) )
        test_split  = train_split + int( ( dataset_size - train_split ) 
            // test_ratio )
        
        # randomly shuffle the indices
        np.random.shuffle( indices )
    
        train_indices = indices[ : train_split ]
        valid_indices = indices[ train_split : test_split ]
        test_indices = indices[ test_split : ] 

        return train_indices, valid_indices, test_indices