"""
mia2 - lda
"""

import numpy as np
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
	plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')

	plt.xlim(x[:, 0].min() - .5, x[:, 0].max() + .5)
	plt.ylim(x[:, 1].min() - .5, x[:, 1].max() + .5)
	plt.xticks(())
	plt.yticks(())

	if plot:
		plt.savefig(plot_path + name + '_raw.png', dpi=150)

	# plot pca
	fig = plt.figure(figsize=(8, 6))
	ax = Axes3D(fig, elev=-150, azim=110)

	ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)

	ax.set_title("First three PCA directions")
	ax.set_xlabel("1st eigenvector")
	ax.w_xaxis.set_ticklabels([])
	ax.set_ylabel("2nd eigenvector")
	ax.w_yaxis.set_ticklabels([])
	ax.set_zlabel("3rd eigenvector")
	ax.w_zaxis.set_ticklabels([])

	if plot:
		plt.savefig(plot_path + name + '_pca.png', dpi=150)


def plot_transformed_data(x, y, plot_path, name, plot=False):
	"""
	plot transformed data lda data points
	"""

	plt.figure(figsize=(8, 6))
	plt.scatter(x[0], x[1], c=y, cmap=plt.cm.Set1, edgecolor='k')
	plt.xlabel('lda component 1')
	plt.ylabel('lda component 2')
	plt.xlim(x[0].min() - .5, x[0].max() + .5)
	plt.ylim(x[1].min() - .5, x[1].max() + .5)
	plt.xticks(())
	plt.yticks(())

	if plot:
		plt.savefig(plot_path + name + '.png', dpi=150)


#------------------------------------------------------------------------------
# Main function
if __name__ == '__main__':

	# some paths
	plot_path = 'ignore/ass6_data/plots/'

	# load iris data
	iris = datasets.load_iris()

	# print some infos
	print("Iris data shape: ", iris.data.shape)
	print("label shape: ", iris.target.shape)
	print("labels: ", np.unique(iris.target))

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
	#plot_iris_data(x, x_pca, y, plot_path, 'iris_data', plot=False)

	# LDA classifier -> in mia lib
	w, bias, x_h, label_list = train_lda_classifier(x, y, method='class_independent', n_lda_dim=1)

	print("transformed data: ", x_h.shape)

	# plot transformed data x_h = [k-1, n]
	plot_transformed_data(x_h, y, plot_path, name='lda_transformed', plot=False)


	# TODO: classify new samples (or the same ones) -> in mia lib
	y_hat = lda_classify(x, w, bias, label_list)


	# TODO: Visualization of new data points compared to transformed data above


	# TODO: confusion matrix check
	#cm = confusion_matrix(y, y_hat)
	#print("confusion matrix:\n", cm)


	# TODO: (maybe nice) calculate accuracies


	# TODO: (optional) comparance with k-means algorithm

	# plot all figures
	plt.show()

