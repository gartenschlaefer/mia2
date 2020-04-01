import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

# my personal mia lib
from mia2 import non_linear_mapping

# Librosa module for calculating the Constant-Q Transform
import librosa as libr

#------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':
    
    # Loading file in memory 
    file_name = 'Cmaj7_9.wav'
    file_path = 'ignore/sounds/'
    full_name = file_path + file_name
     
    audio_data, sampling_rate = libr.load( full_name, sr=None, duration=0.5 )
    non_linear_mapping( )
