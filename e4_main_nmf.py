import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

# mia2 lib
from mia2 import *

import librosa as libr


def plot_signal(x, t, plot_path, name='no_name'):
	"""
	simply plot the time signal
	"""
	plt.figure()
	plt.plot(t, x)
	plt.ylabel("magnitude")
	plt.xlabel("time")
	plt.grid()
	plt.tight_layout()
	plt.savefig(plot_path + name + '_signal.png', dpi=150)
	#plt.show()


def test_scipy_stft(x, fs, N, hop):
	"""
	test stft from scipy...mine is better
	"""
	import scipy.signal

	# scipy stft
	_, _, X = scipy.signal.stft(x, fs=fs, window='hann', nperseg=N, noverlap=hop, nfft=N, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1)
	return X.T


def plot_spec_pca(X, X_pca, plot_path, name='no_name'):
	"""
	plot the spectogram and spectogram of pca
	"""

	# DFT for one frame
	plt.figure()
	plt.plot(X[1])
	plt.ylabel("magnitude")
	plt.xlabel("DFT bins")
	plt.grid()
	plt.tight_layout()
	plt.savefig(plot_path + name + '_dft.png', dpi=150)

	# spectrogram
	plt.figure(figsize=(8, 4))
	plt.imshow(X.T, aspect='auto')
	plt.ylabel("DFT")
	plt.xlabel("frames")
	#plt.tight_layout()
	plt.savefig(plot_path + name + '_spec.png', dpi=150)

	# pca
	plt.figure(figsize=(8, 4))
	plt.imshow(X_pca.T, aspect='auto')
	plt.ylabel("DFT PCA components")
	plt.xlabel("frames")
	#plt.tight_layout()
	plt.savefig(plot_path + name + '_spec_pca.png', dpi=150)
	#plt.show()

#------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':
    
	# file params 
	file_names = ['DL6.wav', 'happy_plug.wav']
	file_name = file_names[0]

	# some paths
	file_path = 'ignore/ass4_data/'
	plot_path = 'ignore/ass4_data/plots/'

	# load file
	x, fs = libr.load(file_path + file_name, sr=11025)

	# --
	# params

	# amount components
	r = 7

	# DFT size
	#N = 1024
	#N = 512
	N = 256

	# hop size
	hop = 128 

	# print some infos
	print("x length: ", len(x))
	print("fames: ", len(x) / hop)

	# time vector
	t = np.arange(0, len(x)/fs, 1/fs)

	# calc stft [m x n] and use magnitude
	X = np.abs(custom_stft(x, N=N, hop=hop, norm=True)[:, :N//2])

	print("X: ", X.shape)

	# pca
	X_pca, ev = calc_pca(X)

	print("PCA: ", X_pca.shape)


	# NMF
	W, H = calc_nmf(X, r=r)

	print("W: ", W.shape)
	print("H: ", H.shape)


	# --
	# some plots
	
	# spec	
	plot_spec_pca(X, X_pca[:, :r], plot_path, name=file_name.split(".")[0])
	
	# plot of signal
	plot_signal(x, t, plot_path, name=file_name.split(".")[0])

	#plt.show()

