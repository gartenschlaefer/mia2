import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

# mia2 lib
from mia2 import *

import librosa as libr

# my personal mia lib
#from mia2 import non_linear_mapping

def plot_signal(x, t, name='no_name'):
	"""
	simply plot the time signal
	"""
	plt.figure()
	plt.plot(t, x)
	plt.grid()
	plt.savefig(plot_path + name + '.png', dpi=150)
	plt.show()


def test_scipy_stft(x, fs, N, hop):
	"""
	test stft from scipy...mine is better
	"""
	import scipy.signal

	# scipy stft
	_, _, X = scipy.signal.stft(x, fs=fs, window='hann', nperseg=N, noverlap=hop, nfft=N, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
	return X.T


#------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':
    
	# file params 
	file_names = ['DL6.wav', 'happy_plug.wav']
	file_name = file_names[1]

	# some paths
	file_path = 'ignore/ass4_data/'
	plot_path = 'ignore/ass4_data/plots/'

	# load file
	x, fs = libr.load(file_path + file_name, sr=11025)

	# DFT size
	#N = 1024
	N = 512

	# hop size
	hop = 128 

	print("x length: ", len(x))
	print("fames: ", len(x) / hop)

	# time vector
	t = np.arange(0, len(x)/fs, 1/fs)

	# calc stft [m x n]
	X = custom_stft(x, N=N, hop=hop, norm=True)[:, :N//2]

	print("X: ", X.shape)

	# checkout PCA
	#X_pca, ev = calc_pca(X)

	#print("PCA: ", X_pca.shape)


	# --
	# some plots

	# plot
	#plt.figure(), plt.plot(np.abs(X[1])), plt.show()

	# some plots
	#plot_signal(x, t, name=file_name.split(".")[0])

