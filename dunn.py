import numpy as np
from scipy.spatial.distance import cdist as distance

def dunnIndex(labels, datapoints, clusters = None):
	points = datapoints.shape[0]
	if clusters is None:
		clusters = np.max(labels)+1
	centroids = np.zeros(shape = (clusters, datapoints.shape[1]))
	for i in range(clusters):
		if datapoints[labels == i, :].shape[0]!=0:
			centroids[i,:] = np.mean(datapoints[labels == i, :], axis = 0)
		
	#distances = np.zeros(shape = (clusters, clusters))
	distancesC = distance(centroids, centroids, metric = 'euclidean')	
	for i in range(clusters):
		distancesC[i][i] = 1000000000
	maxIntraCluster = np.zeros(clusters)
	for i in range(clusters):
		datapointsHere = datapoints[labels == i]
		pointsHere = datapointsHere.shape[0]
		if pointsHere!=0:
			distances = distance(datapointsHere, datapointsHere, metric = 'euclidean')
			maxIntraCluster[i] = np.max(distances)
	
	return np.min(distancesC)/np.max(maxIntraCluster)
