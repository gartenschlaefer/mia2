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
    
    # file name to mat file with onsets and midi notes
    mat_file_name = '01-AchGottundHerr-GTF0s.mat'

    # Loading file in memory 
    file_name = '01-AchGottundHerr_4Kanal.wav'
    file_path = 'ignore/sounds/'
    full_name = file_path + file_name
    audio_data, sampling_rate = libr.load( full_name, sr=None, duration=3 )

    hop = 256
    start_note = 'C2'
    cqt = libr.cqt( audio_data, sr=sampling_rate, hop_length=hop,  
          fmin=libr.note_to_hz( start_note ), n_bins=48, bins_per_octave=12 )
 
    # Define common harmonic structure-----------------------------------------
    # - number of frequency bins is the same as for the cqt -> n_bins = 48
    cqt_bins = 48
    list_chs = [0, 12, 19, 24, 28, 31, 47]

    chs = initial_harmonics( list_chs, np.zeros(( cqt_bins, 1 )), option=2 )
    
    u , v = inverse_filter( cqt, chs, cqt_bins )
    u_bar = non_linear_mapping( u )

    # iterative algorithm------------------------------------------------------
    ( num_rows, num_cols ) = cqt.shape 
    len_harm = len( list_chs ) - 1
    
    # Pre-allocation
    theta_vector = np.zeros( ( len_harm, 1 ), dtype=complex )
    b_vector     = np.zeros( ( len_harm, 1 ), dtype=complex )
    h_bar_vector = np.zeros( ( num_rows, 1 ), dtype=complex )
    u_bar_matrix = np.zeros( ( num_rows, len_harm ), dtype=complex )
    A_matrix     = np.zeros( ( len_harm, len_harm ), dtype=complex )

    # initialise U_bar matrix
    for t in range( num_cols ):
        for out_index, elem in enumerate( list_chs[ 1 : ] ):
            for in_index in range( num_rows ):

                shift = in_index - elem
                if shift < 0: 
                    continue
                elif shift >= 0:
                    u_bar_matrix[ in_index, out_index ] = u_bar[ shift , t]
   
        A_matrix = np.matmul( u_bar_matrix.T, u_bar_matrix  )
        b_vector = np.matmul(( v[: , t] - u_bar[ : , t] ).T, 
                   u_bar_matrix)

        theta_vector = np.matmul( np.linalg.inv(A_matrix), b_vector )

        list_chs = list_chs[ 1 : ]
        for index, elem in  enumerate( list_chs ):
            chs[ elem ] = np.abs( theta_vector[ index ] )

        u , v = inverse_filter( cqt, chs, cqt_bins )
        u_bar = non_linear_mapping( u )

    plot_harmonic_structure( chs ) 

    # Plots--------------------------------------------------------------------
    plot_cqt( cqt, sampling_rate, hop )
    # plot_harmonic_structure( chs ) 
    # plot_pipeline( v, inv_v, u, u_init, u_bar_init )

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
