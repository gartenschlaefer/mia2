import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

# my personal mia lib
from mia2 import non_linear_mapping

# Librosa module for calculating the Constant-Q Transform
import librosa as libr
from librosa.display import specshow

#------------------------------------------------------------------------------
def plot_CQT_spectrum( cqt_spectrum ):
    specshow( libr.amplitude_to_db( np.abs(cqt_spectrum), ref=np.max),
        sr=sampling_rate, x_axis='time', y_axis='cqt_note')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------
def initial_harmonics( list_hormonics, 
    common_harmonic_structure, option=0 ):
    
    if option == 0:
        common_harmonic_structure[ list_hormonics ] = 1
    
    elif (option == 1) or (option == 2):
        for index, elem in enumerate( list_hormonics, 1 ):
            
            if option == 1:
                common_harmonic_structure[ elem ] = 1 / np.sqrt( index )

            if option == 2:
                common_harmonic_structure[ elem ] = 1 / index

    else:
        print( "No other options available!" )
                   
    return common_harmonic_structure    

def plot_harmonic_structure( common_harmonic_structure ):
    
    plt.stem( common_harmonic_structure, linefmt=None, markerfmt=None, 
        basefmt=None, use_line_collection=True)
    plt.xlabel( 'Log-freq. bin number' )
    plt.ylabel( 'Relative amplitude of harmonic component' )
    plt.show()

#------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':
    
    # Loading file in memory 
    file_name = 'Cmaj.wav'
    file_path = 'ignore/sounds/'
    full_name = file_path + file_name
     
    audio_data, sampling_rate = libr.load( full_name, sr=None )

    # Compute and plot CQT-----------------------------------------------------
    # - One frequency bin has a length of 1379 (for Cmaj.wav)
    # - 48 bins in total -> 48 times 1379 
    cqt_spectrum = libr.cqt( audio_data, sr=sampling_rate, fmin=110, 
        n_bins=24, bins_per_octave=12 )
   
    # Define common harmonic structure-----------------------------------------
    # - number of frequency bins is the same as for the cqt -> n_bins = 48
    list_hormonics = [0, 12, 19, 24, 28, 31]
    common_harmonic_structure = np.zeros(( 48, 1 ))

    common_harmonic_structure = initial_harmonics( list_hormonics, 
        common_harmonic_structure, option=2 )
    
    # Plots so far-------------------------------------------------------------
    # plot_CQT_spectrum( cqt_spectrum )
    # plot_harmonic_structure( common_harmonic_structure )
    
    # Initial guess for fundamental frequency distribution---------------------
    # - Done via inverse filter approach.
    

    # Non-linear mapping function----------------------------------------------
    non_linear_mapping( )
