import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

# my personal mia lib
from mia2 import non_linear_mapping
from mia2 import get_onset_mat

# Librosa module for calculating the Constant-Q Transform
import librosa as libr
from librosa.display import specshow

#------------------------------------------------------------------------------
def plot_cqt( cqt, sr, hop ):

    plt.figure(figsize=(8,4))
    specshow( libr.amplitude_to_db( np.abs(cqt), ref=np.max),
        sr=sr, x_axis='time', y_axis='cqt_note', hop_length=hop)
    
    plt.colorbar( format='%+2.0f dB' )
    plt.title( 'Constant-Q power spectrum' )
    plt.tight_layout()
    plt.show()  

#------------------------------------------------------------------------------
def plot_harmonic_structure( chs ):
    plt.stem( chs, linefmt=None, markerfmt=None, 
        basefmt=None, use_line_collection=True)
    plt.xlabel( 'Log-freq. bin number' )
    plt.ylabel( 'Relative amplitude of harmonic component' )
    plt.grid()
    plt.show()

#------------------------------------------------------------------------------
def initial_harmonics( list_chs, 
    chs, option=0 ):
    
    if option == 0:
        chs[ list_chs ] = 1
    
    elif (option == 1) or (option == 2):
        for index, elem in enumerate( list_chs, 1 ):
            
            if option == 1:
                chs[ elem ] = 1 / np.sqrt( index )

            if option == 2:
                chs[ elem ] = 1 / index

    else:
        print( "No other options available!" )
    
    return chs  

#------------------------------------------------------------------------------
def inverse_filter( cqt, chs, fft_bins ):
    
    v = np.power( np.abs( cqt ), 2 )
    
    inv_v  = ifft( v, fft_bins, axis=0 )
    inv_chs = np.conj( ifft( chs, fft_bins, axis=0 ))
    
    u = fft( np.multiply( inv_v, inv_chs ), axis=0 )

    return u , v 

#------------------------------------------------------------------------------
def plot_pipeline( inv_observed_spectrum, inv_chs, 
    estimate_freq_distro, u, u_bar ):
  """
  plot pipeline of the algorithm
  """
  s_frame = 0
  t_frame = 1000

  plt.figure()
  plt.imshow(np.abs(inv_observed_spectrum[:, s_frame:t_frame]), aspect='auto')
  plt.title("inv observed spectrum")
  plt.ylabel("time (specmurt)")
  plt.xlabel("frames")
  #plt.show()

  plt.figure()
  plt.plot(np.abs(inv_chs))
  plt.title("inv harm struct")
  plt.ylabel("time (specmurt)")
  plt.xlabel("frames")
  #plt.show()

  plt.figure()
  plt.imshow(np.abs(estimate_freq_distro[:, s_frame:t_frame]), aspect='auto')
  plt.title("V / H")
  plt.ylabel("time (specmurt)")
  plt.xlabel("frames")
  #plt.show()

  plt.figure()
  plt.imshow(np.abs(u[:, s_frame:t_frame]), aspect='auto')
  plt.title("u = fft(V / H)")
  plt.ylabel("frequency")
  plt.xlabel("frames")
  #plt.show()

  plt.figure()
  plt.imshow(np.abs(u_bar[:, s_frame:t_frame]), aspect='auto')
  plt.title("u bar")
  plt.ylabel("frequency")
  plt.xlabel("frames")
  plt.show()

#------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':
    
    # file name to mat file with onsets and midi notes-------------------------
    mat_file_name = '01-AchGottundHerr-GTF0s.mat'

    # Loading file in memory---------------------------------------------------
    file_name = 'Cmaj7_9.wav'
    file_path = 'ignore/sounds/'
    full_name = file_path + file_name
    audio_data, sampling_rate = libr.load( full_name, sr=None )

    # CQT Params---------------------------------------------------------------
    hop = 256
    start_note = 'C2'
    cqt = libr.cqt( audio_data, sr=sampling_rate, hop_length=hop,  
          fmin=libr.note_to_hz( start_note ), n_bins=48, bins_per_octave=12 )
 
    # Define common harmonic structure-----------------------------------------
    cqt_bins = 48
    list_chs = [ 0, 12, 19, 24, 28, 31 ]

    # First initialisation of fundamental frequency distribution---------------
    chs   = initial_harmonics( list_chs, np.zeros(( cqt_bins, 1 )), option=1 )
    u, v  = inverse_filter( cqt, chs, cqt_bins )
    u_bar = non_linear_mapping( u )
    len_u = len( u_bar[ : , 0 ] )

    # iterative algorithm------------------------------------------------------
    ( num_rows, num_cols ) = cqt.shape 
    len_harm = len( list_chs ) - 1
    
    b_matrix = np.zeros(( len_harm, num_cols ), dtype=complex)
    t_matrix = np.zeros(( len_harm, num_cols ), dtype=complex)
    u_bar_matrix = np.zeros(( len_harm, num_rows ), dtype=complex)

    for i in range( num_cols ):
        for j, elem_j in enumerate( list_chs[ 1: ] ):
            u_bar_matrix[ j, elem_j : ] = u_bar[ 0 : len_u - elem_j , j ]     

        A_matrix = u_bar_matrix @ u_bar_matrix.T 
        inv_A_matrix = np.linalg.inv( A_matrix )
        
        b_matrix[ : , i ] = u_bar_matrix @ ( v[ : , i ] - u_bar[ : , i ] )
        t_matrix[ : , i ] = inv_A_matrix @ b_matrix[ : , j ] 

        for k in list_chs[ 1: ]:
            chs[ k ] = np.abs( np.amax(  t_matrix[ : , i ] ))
        
        u , v = inverse_filter( cqt, chs, cqt_bins )

        print( 'Number of iterations: {}'.format( i ) )
        tolerance = 0.1
        if i == 0:
            u_pre_iter = u;
        else:
            update_amount = np.linalg.norm( ( u - u_pre_iter ) )
            if update_amount <= tolerance:
                break
            else:
                continue
            u_pre_iter = u
            
        u_bar = non_linear_mapping( u )

    # Plots--------------------------------------------------------------------
    # plot_cqt( cqt, sampling_rate, hop )
    # plot_harmonic_structure( chs ) 
    # plot_pipeline( v, inv_chs, u, u, u_bar )
    plt.plot( libr.cqt_frequencies(48, fmin=libr.note_to_hz('C2'), 
                bins_per_octave=12 ), np.abs( u_bar[ : , 300] ))
    # plt.xlabel( 'Frequency log-scale' )
    # plt.ylabel( 'Fundamental frequency distribution' )
    plt.show()

    # get the onsets and midi notes of the audiofile---------------------------
    onsets, m, t = get_onset_mat( file_path + mat_file_name )

    # delta t of onsets -> 0.01s
    dt = np.around(t[1] - t[0], decimals=6)

    # get hop size of onsets
    hop_onset = sampling_rate * dt

    # print("delta t: ", dt)
    # print("onsets: ", onsets.shape)
    # print("midi: ", m.shape)

    # plt.figure()
    # plt.plot(t, m.T)
    #plt.show()

    # plt.figure()
    # plt.plot(np.abs(u_bar[:, 100:110]))
    # plt.title("u bar")
    # plt.ylabel("Magnitude")
    # plt.xlabel("frequency")
    # plt.show()

    # plt.figure()
    # plt.imshow(np.abs(u_bar_init), aspect='auto', extent=[0, t[-1], 
    #     libr.core.note_to_midi('C6'), libr.core.note_to_midi('C2')])
    # 
    # plt.title("u bar")
    # plt.ylabel("midi note")
    # plt.xlabel("frames")
    # plt.show()
