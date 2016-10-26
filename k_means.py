import numpy as np
import time

debug = False


def k_means_genetic_generation(solution_set, datapoints, k, dim, points, top_n, crossover_points, p, allowed_error):
	new_solution = np.zeros(shape = solution_set.shape, dtype = np.int32)	
	scores = np.zeros(shape = (solution_set.shape[0]))
	for i in range(solution_set.shape[0]):
		solution = np.copy(solution_set[i])
		centres = findCentres(datapoints, np.copy(solution), k, dim)
		scores[i] = calculateError(datapoints, np.copy(solution), np.copy(centres), k , dim)
#	print(scores, np.argmin(scores),scores.argsort()[0])
	top_n_indices = scores.argsort()[:top_n]
	if scores[top_n_indices[0]]<allowed_error:
		return (True, solution_set, top_n_indices[0], scores[top_n_indices[0]]) 
	for i in range(top_n):
		new_solution[i] = np.copy(solution_set[top_n_indices[i]][:])
	
	crossover_considerations = np.zeros(shape = (solution_set.shape[0] - top_n, points))
	top_n_indices = scores.argsort()[top_n:]
		
	for i in range(solution_set.shape[0] - top_n):
		crossover_considerations[i] = np.copy(solution_set[top_n_indices[i]])
	
	final_crossover = crossovers(np.copy(crossover_considerations), k, dim, points, int((solution_set.shape[0] - top_n)/2), n = crossover_points)
	mutated = mutations(np.copy(final_crossover), k, dim, points, p)
	top_n_indices = scores.argsort()[:top_n]
	for i in range(solution_set.shape[0] - top_n):
		new_solution[top_n + i][:] = np.copy(mutated[i])
	return (False, np.copy(new_solution), top_n_indices[0], scores[top_n_indices[0]])
	

def compare(array1, array2):
	if array1.shape != array2.shape:
		return False
	else:
		for i in range(array1.shape[0]):
			#for j in range(array1.shape[1]):
			if array1[i]!=array2[i]:
				return False
		return True

def k_means_genetic(datapoints, dim, points, k, solution_set_size = 100, top_n = 30, crossover_points = 2, p = 0.1, allowed_error = 10, max_generations = 10):
	solution_set = np.random.randint(0, k, size = (solution_set_size, points))
	debug2 = False
	debug2 and print(solution_set)
	generation = 0
	debug and print("Generation", generation, "starting")
	x = None
	while True:
		x = k_means_genetic_generation(np.copy(solution_set), datapoints, k, dim, points, top_n, crossover_points, p, allowed_error)
		if x[0]:
			break
		else:
			solution_set = np.copy(x[1])
			print(x[3], x[2])
		print(solution_set)
		generation+=1
		if generation>max_generations:
			break			
	print(x[1][x[2]])
	print("Error = ", x[2])

def crossovers(solution_set, k, dim, points, m, n = 2):
	""" 2*m indicates the number of solutions that need to be generated from the crossover.
	n is the number of crossovers"""
	
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
		new_solutions[2*i][:] = np.copy(gene_1_new.astype(np.int32, copy = False))
		new_solutions[2*i+1][:] = np.copy(gene_2_new.astype(np.int32, copy = False))
	return np.copy(new_solutions)


def mutations(solution_set, k, dim, points, p = 0.1):
	for i in range(solution_set.shape[0]):
		if np.random.rand()<=p:
			debug and print("Yes")
			position = np.random.randint(0, points)
			debug and print("Position is ", position)			
			solution_set[i][position] = np.random.randint(0,k)
	return np.copy(solution_set)


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
	return np.copy(centres)

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
	return np.copy(temp3)

def k_means_regular(datapoints, points, dim, k, allowed_error = 10, max_generations = 10):
	solution = np.random.randint(k, size = points)
	centres = findCentres(datapoints, solution, k, dim)
	error = calculateError(datapoints, solution, centres, k, dim)
	generation = 0
	while error>allowed_error and generation<max_generations:
		solution = reassignPoints(datapoints.copy(), centres.copy(), points, k, dim)
		centres = findCentres(datapoints.copy(), solution.copy(), k, dim)
		error = calculateError(datapoints.copy(), solution.copy(), centres.copy(), k, dim)
		generation+=1
		print(generation, error)
	return (solution, error, generation)				

#Initial data which is provided
k = 10
points = 40000
dim = 91
datapoints = np.random.rand(points, dim)
print(k_means_regular(datapoints, points, dim, k, allowed_error = 21.5))
input()
print(k_means_genetic(datapoints, dim, points, k, solution_set_size = 100, top_n = 2, crossover_points = 2, p = 0.15, allowed_error = 21.5))

