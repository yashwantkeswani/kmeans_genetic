import numpy as np
import timeit

from sklearn.cluster import KMeans

debug = False

def crossovers_centres(solution_set, k, dim, points, m, n = 2):
	""" 2*m indicates the number of solutions that need to be generated from the crossover.
	n is the number of crossovers"""
	
	if (2*m)>solution_set.shape[0]:
		print("Error!")
		return
	combinations = np.linspace(0,solution_set.shape[0]-1,solution_set.shape[0], dtype = int)
	np.random.shuffle(combinations)
	combinations = combinations[:2*m]
	debug and print("Combinations", combinations)
	new_solutions = np.zeros(shape = (2*m, k, dim))
	for i in range(m):
		gene1 = solution_set[combinations[2*i]].reshape(k*dim)	
		gene2 = solution_set[combinations[2*i+1]].reshape(k*dim)
		crossover_points = np.sort(np.random.randint(1, k*dim, size = n))
		gene_1_new = np.zeros(shape = (1, k*dim))
		gene_2_new = np.zeros(shape = (1, k*dim))
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
		new_solutions[2*i] = np.copy(gene_1_new[0].reshape(k, dim))
		new_solutions[2*i+1] = np.copy(gene_2_new[0].reshape(k, dim))
	return np.copy(new_solutions)


def mutations_centres(solution_set, k, dim, points, p = 0.1):
	for i in range(solution_set.shape[0]):
		if np.random.rand()<=p:
			debug and print("Yes")
			position = np.random.randint(0, k)
			position2 = np.random.randint(0, dim)
			debug and print("Position is ", position)		
			solution_set[i][position][position2] = np.random.rand()
	return np.copy(solution_set)

def k_means_genetic_generation_centres(solution_set, datapoints, k, dim, points, top_n, crossover_points, p, allowed_error):
	new_solution = np.zeros(shape = solution_set.shape)	
	scores = np.zeros(shape = (solution_set.shape[0]))
	for i in range(solution_set.shape[0]):
		centres = np.copy(solution_set[i])
		clusters = 	reassignPoints(datapoints, np.copy(centres), points, k, dim)
		scores[i] = calculateError(datapoints, np.copy(clusters), np.copy(centres), k , dim)		

	top_n_indices = scores.argsort()[:top_n]
	if scores[top_n_indices[0]]<allowed_error:
		return (True, solution_set, top_n_indices[0], scores[top_n_indices[0]]) 

	for i in range(top_n):
		new_solution[i] = np.copy(solution_set[top_n_indices[i]])
	
	crossover_considerations = np.zeros(shape = (solution_set.shape[0] - top_n, k, dim))
	top_n_indices = scores.argsort()[top_n:]
	for i in range(solution_set.shape[0] - top_n):
		crossover_considerations[i] = np.copy(solution_set[top_n_indices[i]])
	final_crossover = crossovers_centres(np.copy(crossover_considerations), k, dim, points, int((solution_set.shape[0] - top_n)/2), n = crossover_points)		
	mutated = mutations_centres(np.copy(final_crossover), k, dim, points, p)
	top_n_indices = scores.argsort()[:top_n]
	for i in range(solution_set.shape[0] - top_n):
		new_solution[top_n + i] = np.copy(mutated[i])
	return (False, np.copy(new_solution), top_n_indices[0], scores[top_n_indices[0]])




def k_means_genetic_centres(datapoints, dim, points, k, solution_set_size = 100, top_n = 30, crossover_points = 2, p = 0.1, allowed_error = 10, max_generations = 10):
	generation = 0
	solution_set = np.random.rand(solution_set_size, k, dim)	
	while True:
		x = k_means_genetic_generation_centres(np.copy(solution_set), datapoints, k, dim, points, top_n, crossover_points, p, allowed_error)
		if x[0]:
			break
		else:
			solution_set = np.copy(x[1])
			print(x[3], x[2])
		debug and print(solution_set)
		generation+=1
		if generation>max_generations:
			break			
	print(x[1][x[2]])
	print("Error = ", x[2])
		

def k_means_genetic_generation(solution_set, datapoints, k, dim, points, top_n, new_n, crossover_points, p, allowed_error):
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
		new_solution[i] = np.copy(solution_set[top_n_indices[i]])
	
	new_solution[top_n:top_n+new_n, :] = np.random.randint(0, k, size = (new_n, points))	
	crossover_considerations = np.zeros(shape = (solution_set.shape[0] - top_n - new_n, points))
	top_n_indices = scores.argsort()[top_n:]
		
	for i in range(solution_set.shape[0] - top_n - new_n):
		crossover_considerations[i] = np.copy(solution_set[top_n_indices[i]])
	
	final_crossover = crossovers(np.copy(crossover_considerations), k, dim, points, int((solution_set.shape[0] - top_n - new_n)/2), n = crossover_points)
	mutated = mutations(np.copy(final_crossover), k, dim, points, p)
	top_n_indices = scores.argsort()[:top_n]
	for i in range(solution_set.shape[0] - top_n - new_n):
		new_solution[top_n + new_n + i] = np.copy(mutated[i])
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

def k_means_genetic(datapoints, dim, points, k, solution_set_size = 100, top_n = 30, new_n = 40, crossover_points = 1, p = 0.1, allowed_error = 10, max_generations = 10):
	solution_set = np.random.randint(0, k, size = (solution_set_size, points))
	debug2 = False
	debug2 and print(solution_set)
	generation = 0
	debug and print("Generation", generation, "starting")
	x = None
	while True:
		x = k_means_genetic_generation(np.copy(solution_set), datapoints, k, dim, points, top_n, new_n, crossover_points, p, allowed_error)
		if x[0]:
			break
		else:
			print(x[1])
			solution_set = np.copy(x[1])
		debug and print(solution_set)
		generation+=1
		if generation>max_generations:
			break			
	return x

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


def mutations(solution_set, k, dim, points, p = 0.1, number_in_each = None):
	master_array = np.arange(points)	
	if number_in_each is None:
		number_in_each = int(points*0.01)
	for i in range(solution_set.shape[0]):
		if np.random.rand()<=p:
			debug and print("Yes")
			np.random.shuffle(master_array)
			position = np.random.choice(master_array, number_in_each)
			debug and print("Position is ", position)
			for each in position:
				solution_set[i][each] = np.random.randint(0,k)
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
		error += np.sum(np.sqrt(np.sum(temp, axis = 1)), axis = 0)
	return error

def reassignPoints(datapoints, centres, points, k, dim):
	distance = np.zeros(shape=(points, k))
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
#		print(generation, error)
	return (solution, error, generation)				


def paper_approach(datapoints, dim, points, k, solution_set_size = 100, top_n = 30, new_n = 40, crossover_points = 1, p = 0.1, allowed_error = 10, max_generations = 10, initial_iteration = 2):
	"""http://ir.kaist.ac.kr/anthology/2009.06-Al-Shboul.pdf"""
	solution_set = np.zeros(shape = (solution_set_size, points))
	for i in range(solution_set_size):
		solution_set[i] = k_means_regular(datapoints, points, dim, k, allowed_error = 0.001, max_generations = initial_iteration)[0]
	generation = 0
	x = None
	while True:
		x = k_means_genetic_generation(np.copy(solution_set), datapoints, k, dim, points, top_n, new_n, crossover_points, p, allowed_error)
		if x[0]:
			break
		else:
			#print(x[1])
			solution_set = np.copy(x[1])
#			print(x[3], x[2])
		debug and print(solution_set)
		generation+=1
		if generation>max_generations:
			break			
#	print(x[1][x[2]])
	print("Error = ", x[3])
	return x
	
#Initial data which is provided
k = 15
points = 1000
dim = 10
datapoints = np.random.rand(points, dim)
s = k_means_regular(datapoints, points, dim, k, allowed_error = 0.001, max_generations = 500)
print(s[1])
print(k_means_regular(datapoints, points, dim, k, allowed_error = 0.001, max_generations = 1)[1])
print(paper_approach(datapoints, dim, points, k, solution_set_size = 50, top_n = 20, new_n = 0, crossover_points = 2, p = 0.001, allowed_error = 21.5, max_generations = 20, initial_iteration = 5)[1])
#input()
#print(k_means_genetic(datapoints, dim, points, k, solution_set_size = 100, top_n = 10, new_n = 10, crossover_points = 2, p = 1, allowed_error = 21.5, max_generations = 100))

