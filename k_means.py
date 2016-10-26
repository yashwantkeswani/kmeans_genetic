import numpy as np
import time

def findCentres(datapoints, solution, k, dim):
	centres = np.random.rand(k, dim)
	total = 0
	for i in range(k):
		temp = (datapoints[solution==i,:])
		total += temp.size/dim
		if temp.size>0:
			centres[i,:] = np.mean(temp, axis = 0)
		else:
			centres[i,:] = np.random.randint(1, size = dim)
	return centres

def calculateError(datapoints, solution, centres, k , dim):
	error = 0.0
	for i in range(k):
		temp = (datapoints[solution == i, :] - centres[i,:])
		temp = np.multiply(temp, temp)
		error += np.sum(np.sum(temp, axis = 1), axis = 0)
	return np.sqrt(error)

def reassignPoints(datapoints, centres, points, k, dim):
	distance = np.random.rand(points, k)
	for i in range(k):
		distance[:,i] = np.sqrt(np.sum(np.multiply(datapoints - centres[i,:],datapoints - centres[i,:]), axis=1))
	temp3 = np.argmin(distance, axis = 1)
	return temp3.copy()

def k_means_regular(datapoints, points, dim, k, allowed_error = 10):
	count = 0	
	solution = np.random.randint(k, size = points)
	centres = findCentres(datapoints, solution, k, dim)
	error = calculateError(datapoints, solution, centres, k, dim)
	while error>allowed_error:
		solution = reassignPoints(datapoints.copy(), centres.copy(), points, k, dim)
		centres = findCentres(datapoints.copy(), solution.copy(), k, dim)
		error = calculateError(datapoints.copy(), solution.copy(), centres.copy(), k, dim)
		count+=1
		print(count, error)
	return (solution, error, count)				

#Initial data which is provided
k = 100
points = 40000
dim = 91
datapoints = np.random.rand(points, dim)

print(k_means_regular(datapoints, points, dim, k, allowed_error = 10))
