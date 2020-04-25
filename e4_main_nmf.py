import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft

# mia2 lib
from mia2 import custom_stft, calc_pca, calc_nmf, time_shift

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
	_, _, X = scipy.signal.stft(x, 
		fs=fs, window='hann', nperseg=N, noverlap=hop, nfft=N, detrend=False, 
		return_onesided=True, boundary='zeros', padded=True, axis=-1 )
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


def plot_nmf_wh(W, H, d, r, max_iter, plot_path, name='no_name'):
	"""
	plot NMF matrices W and H
	"""

	# W
	plt.figure()
	plt.imshow(W, cmap='cividis', aspect='auto')
	plt.ylabel("DFT")
	plt.xlabel("Columns of W")

	ax = plt.gca()
	ax.set_xticks(np.arange(0, r, 1))
	ax.set_xticklabels(np.arange(0, r, 1))
	plt.savefig("{}{}_nmf-W_d-{:.4f}_r-{}_it-{}".format(plot_path, name, d, r, max_iter).replace(".", "p") + '.png', dpi=150)

	# H
	plt.figure(figsize=(8, 4))
	plt.imshow(H, cmap='magma', aspect='auto')
	plt.ylabel("Rows of H")
	plt.xlabel("frames")

	ax = plt.gca()
	ax.set_yticks(np.arange(0, r, 1))
	ax.set_yticklabels(np.arange(0, r, 1))
	plt.savefig("{}{}_nmf-H_d-{:.4f}_r-{}_it-{}".format(plot_path, name, d, r, max_iter).replace(".", "p") + '.png', dpi=150)


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
	x, fs = libr.load( file_path + file_name, sr=11025 )

	# --
	# params

	# amount components
	r = 4

	# number of max iterations
	max_iter = 1000

	# DFT size
	#N = 1024
	#N = 512
	N = 256

	# hop size
	hop = 128 

	# print some infos
	print("x length: ", len(x))
	print("fames: ", len(x) // hop)

	# time vector
	t = np.arange(0, len(x)/fs, 1/fs)

	# calc stft [m x n] and use magnitude
	X = np.abs( custom_stft(x, N=N, hop=hop, norm=True)[:, :N//2] )
	print("X: ", X.shape)

	# pca
	X_pca, ev = calc_pca(X)
	print("PCA: ", X_pca.shape)

	# NMF with lee seung algorithm
	W, H, d = calc_nmf(X.T, R=r, T=10, algorithm='smaragdis', max_iter=max_iter)

	print("W: ", W.shape)
	print("H: ", H.shape)

	# --
	# some plots
	
	# nmf	
	# plot_nmf_wh(W, H, d, r, max_iter, plot_path, name=file_name.split(".")[0])
	
	# spec	
	#plot_spec_pca(X, X_pca[:, :r], plot_path, name=file_name.split(".")[0])
	
	# plot of signal
	# plot_signal(x, t, plot_path, name=file_name.split(".")[0])

	# plt.show()

