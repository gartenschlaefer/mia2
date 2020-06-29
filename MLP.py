#---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

#---------------------------------------------------------------
class MLP_Net( nn.Module ):
    def __init__( self, in_dim, hid_dim, out_dim ):
        super( MLP_Net, self ).__init__(  )
        self.fc1 = nn.Linear( in_dim, hid_dim )
        self.fc2 = nn.Linear( hid_dim, out_dim )
    
    def forward( self, x ):
        x = torch.relu( self.fc1( x ) )
        x = torch.sigmoid( self.fc2( x ) )
        return x