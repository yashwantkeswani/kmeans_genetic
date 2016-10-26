import numpy as np
import time

debug = False

def crossovers(solution_set, k, dim, m, n = 2):
	""" 2*m indicates the number of solutions that need to be generated from the crossover.
	n is the number of genes that participate in the crossover"""
	points = dim
	if (2*m)>solution_set.shape[0]:
		print("Error!")
		return
	combinations = np.linspace(0,solution_set.shape[0]-1,solution_set.shape[0], dtype = int)
	np.random.shuffle(combinations)
	combinations = combinations[:2*m]
	debug and print("Combinations", combinations)
	new_solutions = np.zeros(shape = (2*m, points))
	for i in range(m):
		gene1 = solution_set[combinations[2*i]]			
		gene2 = solution_set[combinations[2*i+1]]			
		crossover_points = np.sort(np.random.randint(1, points-1, size = n))
		gene_1_new = np.zeros(shape = (1,points))
		gene_2_new = np.zeros(shape = (1,points))
		debug and print("Crossover", crossover_points)
		gene_1_new[0][:crossover_points[0]] = gene2[:crossover_points[0]]
		gene_2_new[0][:crossover_points[0]] = gene1[:crossover_points[0]]
		for j in range(1,n):
			if j%2==0:
				gene_2_new[0][crossover_points[j-1]:crossover_points[j]] = gene1[crossover_points[j-1]:crossover_points[j]]
				gene_1_new[0][crossover_points[j-1]:crossover_points[j]] = gene2[crossover_points[j-1]:crossover_points[j]]
			else:
				gene_1_new[0][crossover_points[j-1]:crossover_points[j]] = gene1[crossover_points[j-1]:crossover_points[j]]
				gene_2_new[0][crossover_points[j-1]:crossover_points[j]] = gene2[crossover_points[j-1]:crossover_points[j]]

		if n==1:
			gene_2_new[0][crossover_points[-1]:] = gene2[crossover_points[-1]:]
			gene_1_new[0][crossover_points[-1]:] = gene1[crossover_points[-1]:]
		else:		
			if n%2==1:
				gene_2_new[0][crossover_points[-1]:] = gene1[crossover_points[-1]:]
				gene_1_new[0][crossover_points[-1]:] = gene2[crossover_points[-1]:]
			else:
				gene_2_new[0][crossover_points[-1]:] = gene1[crossover_points[-1]:]
				gene_1_new[0][crossover_points[-1]:] = gene2[crossover_points[-1]:]
		debug and print("gene1", gene1)
		debug and print("gene1", gene2)
		debug and print("gene1_u", gene_1_new)
		debug and print("gene2_u", gene_2_new)
		new_solutions[2*i][:] = gene_1_new
		new_solutions[2*i+1][:] = gene_2_new
	return new_solutions.copy()


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

#print(k_means_regular(datapoints, points, dim, k, allowed_error = 10))

s = np.random.randint(0,5,size = (6,6))
z = crossovers(s, 5, 6, 3, n = 1)
print(z)
