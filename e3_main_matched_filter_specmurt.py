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
# Main function
if __name__ == '__main__':
    
    # Loading file in memory 
    file_name = 'Cmaj7_9.wav'
    file_path = 'ignore/sounds/'
    full_name = file_path + file_name
     
    audio_data, sampling_rate = libr.load( full_name, sr=None )

    # Compute and plot CQT
    cqt_spectrum = libr.cqt( audio_data, sr=sampling_rate, fmin=50, 
        n_bins=120, bins_per_octave=24 )
    
    plot_CQT_spectrum( cqt_spectrum )
    
    #--------------------------------------------------------------------------
    non_linear_mapping( )
