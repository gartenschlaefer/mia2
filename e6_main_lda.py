"""
mia2 - lda
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.metrics import confusion_matrix

# mia2 lib
from mia2 import calc_pca, train_lda_classifier, lda_classify

def plot_iris_data(x, x_pca, y, plot_path, name, plot=False):
	"""
	visualization of iris data, adapted from
	https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
	"""

	plt.figure(figsize=(8, 6))

	# Plot the training points
	plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Blues, edgecolor='k')
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(x[:, 0].min() - .5, x[:, 0].max() + .5)
	plt.ylim(x[:, 1].min() - .5, x[:, 1].max() + .5)
	#plt.xticks(())
	#plt.yticks(())

	if plot:
		plt.savefig(plot_path + name + '_raw.png', dpi=150)

	# pca 2d
	plt.figure(figsize=(8, 6))
	plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap=plt.cm.Blues, edgecolor='k')
	plt.xlabel('pca component 1')
	plt.ylabel('pca component 2')
	plt.xlim(x_pca[:, 0].min() - .5, x_pca[:, 0].max() + .5)
	plt.ylim(x_pca[:, 1].min() - .5, x_pca[:, 1].max() + .5)

	if plot:
		plt.savefig(plot_path + name + '_pca-2d.png', dpi=150)

	# pca 3d
	fig = plt.figure( figsize=(8, 6) )
	ax = Axes3D( fig, elev=-150, azim=110 )

	ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=y, cmap=plt.cm.Blues, edgecolor='k', s=40)

	ax.set_xlabel("pca component 1")
	ax.w_xaxis.set_ticklabels([])
	
	ax.set_ylabel("pca component 2")
	ax.w_yaxis.set_ticklabels([])

	ax.set_zlabel("pca component 3")
	ax.w_zaxis.set_ticklabels([])

	if plot:
		plt.savefig(plot_path + name + '_pca.png', dpi=150)

def computing_boundaries( x, mu_k_h ):

	n,m = x.shape

	mu_12 = ( mu_k_h[ : , 0 ]  +  mu_k_h[ : , 1 ] ) / 2
	mu_23 = ( mu_k_h[ : , 1 ]  +  mu_k_h[ : , 2 ] ) / 2
	mu_31 = ( mu_k_h[ : , 2 ]  +  mu_k_h[ : , 0 ] ) / 2
	mu = np.array( [ mu_12, mu_23, mu_31 ] ).T

	# calculation of distances:
	distance_1_2 = mu_k_h[ : , 0 ]  +  mu_k_h[ : , 1 ]
	distance_2_3 = mu_k_h[ : , 1 ]  +  mu_k_h[ : , 2 ]

	slope_1 = -1 * ( distance_1_2[1] / distance_1_2[0] )
	slope_2 = -1 * ( distance_2_3[1] / distance_2_3[0] )

	intercept_1 = mu_12[1] - slope_1 * mu_12[0]
	intercept_2 = mu_23[1] - slope_2 * mu_23[0]
	
	boundary_1 = ( slope_1 * x[0, :] ) + intercept_1
	boundary_2 = ( slope_2 * x[0, :] ) + intercept_2

	# new labels:--------------------------------------------------------------
	classes = [ 0 , 1 , 2 ]
	
	test_1 = np.array( [ x[0 , :], boundary_1 ] )
	test_2 = np.array( [ x[0 , :], boundary_2 ] )

	b_1 = x[1 , :] < test_1[1 , :]
	b_2 = np.logical_and( x[1 , :] >= test_1[1 , :], x[1 , :] >= test_2[1 , :] )
	b_3 = x[1 , :] < test_2[1 , :]

	X = np.zeros((  m , 1 ))
	X[ b_1 ] = classes[ 0 ]
	X[ b_2 ] = classes[ 1 ]
	X[ b_3 ] = classes[ 2 ]

	return mu, X, boundary_1, boundary_2

def plot_transformed_data( x_h, mu_k_h, y, plot_path, mu, y_hat ,boundary_1, 
		boundary_2, name, plot=False ):
	"""
	plot transformed data lda data points
	"""

	#--------------------------------------------------------------------------
	plt.figure( figsize=(8, 6) )
	plt.scatter( x_h[0], x_h[1], c=y, cmap=plt.cm.Blues, edgecolor='k' )
	plt.scatter( mu_k_h[ 0 , : ], mu_k_h[ 1 , : ], color='k', marker='x', 
		antialiased=True )

	# Little helper to get the individual colors of the colormap
	# cmap = matplotlib.cm.get_cmap( 'Blues' )
	# rgba = cmap( 0.99 )
	# print( rgba )

	# Plot class Separation threshholds:---------------------------------------
	plt.scatter( mu[ 0 , : ], mu[ 1 , : ], color='k', s=10 )

	plt.plot( x_h[0, :], boundary_1, '--', 
	color=( 0.1271049596309112, 0.4401845444059977, 0.7074971164936563, 1.0 ))
	
	plt.plot( x_h[0, :], boundary_2 ,'--',
	color=( 0.03137254901960784, 0.19635524798154555, 0.4316647443291042, 1.0 ))	
	
	plt.xlabel( 'lda component 1' )
	plt.ylabel( 'lda component 2' )
	
	plt.xlim( x_h[0].min() - .5, x_h[0].max() + .5 )
	plt.ylim( x_h[1].min() - .5, x_h[1].max() + .5 )
	plt.grid( True )

	if plot:
		plt.savefig( plot_path + name + '.png', dpi=150 )

def plot_confusion_matrix( cm ):

	predict_labels = [ "setosa", "veriscolor", "virginica" ]
	true_lables = [ "setosa", "veriscolor", "virginica" ]

	fig, ax = plt.subplots( figsize=(8, 6) )
	im = ax.imshow( cm, cmap=plt.cm.Blues )

	for predict in range( len( predict_labels )):
		for true in range( len(true_lables )):
			if predict != true:
				font_color = 'black'
			else:
				font_color = 'white'

			text = ax.text( true, predict, cm[ predict, true ], 
					ha='center', va='center', color=font_color )

	ax.set_xticks( np.arange( len( predict_labels )))
	ax.set_yticks( np.arange( len( true_lables )))

	ax.set_xticklabels( predict_labels )
	ax.set_yticklabels( true_lables )

	plt.xlabel( 'Predicted Labels' )
	plt.ylabel( 'True Labels' )

#------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':

	# some paths
	plot_path = 'ignore/ass6_data/plots/'

	# load iris data
	iris = datasets.load_iris()

	# print some infos
	print( "Iris data shape: ", iris.data.shape )
	print( "label shape: ", iris.target.shape )
	print( "labels: ", np.unique( iris.target ))

	# use n features (max are 4), TODO: remove later for testing whole feature set!
	n_features = 4

	# convention:
	# n sampels
	# m features

	# extract data [n x m] and corresponding labels
	x = iris.data[:, :n_features] 
	y = iris.target
	
	# apply pca
	x_pca, _ = calc_pca(iris.data)
	print("pca: ", x_pca.shape)

	# visualize iris data
	# plot_iris_data(x, x_pca, y, plot_path, 'iris_data', plot=False)

	# LDA classifier -> in mia lib
	w, bias, x_h, mu_k_h, label_list = train_lda_classifier( x, y, 
		method='class_independent', n_lda_dim=1 )
	print( "transformed data: ", x_h.shape )

	# TODO: classify new samples (or the same ones) -> in mia lib
	mu, y_hat ,boundary_1, boundary_2 = computing_boundaries( x_h, mu_k_h )

	# plot transformed data x_h = [k-1, n]
	plot_transformed_data( x_h, mu_k_h, y, plot_path, mu, y_hat, boundary_1, 
		boundary_2, name='lda_transformed', plot=False )

	# TODO: confusion matrix check
	cm = confusion_matrix( y, y_hat )
	print( "confusion matrix:\n", cm )
	# plot_confusion_matrix( cm )

	# TODO: (optional) comparance with k-means algorithm

	# plot all figures
	plt.show()
