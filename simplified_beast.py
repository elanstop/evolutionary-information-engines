#from __future__ import division
import matplotlib.pyplot as plt
from random import random, randrange, uniform
from math import exp, ceil
import numpy as np
from numpy import matmul
from numpy import linalg as LA
import time
from math import log, tan
import cmath
import pickle

#we evolve information engines to get better and better at extracting work from the correlations
#in a bit string consisting of alternating 1s and 0s. The information engines (aka bitbeasts) are
#described by 4x4 transition matrices that satisfy detailed balance.

#the transition matrices give the probability of transitioning from each of four, joint {beast state}x{tape state}
#states to each other during an interaction interval.

#full description of the problem can be found at https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.042115

#upper bound on matrix entries
max_entry = 100

#interaction timescale
tau = 1

#any large number, used to approximate the matrix exponential
n = 10**(9)

#bounds for the mutation operator: old values multiplied by exp(gamma*tan(theta))
#where theta is sampled uniformly from (theta_min,theta_max)
theta_min = -3.14/4
theta_max = 3.14/4
gamma = 7

#do we require generated matrices to be embeddable, or merely to be reversible?
embeddable = False

I = np.identity(4)

#the matrix that models feeding a 1 from the input tape to the beast
F1 = np.matrix([[0,0,0,0],[1,1,0,0],[0,0,0,0],[0,0,1,1]])

#the matrix that models feeding a 0 from the input tape to the beast
F0 = np.matrix([[1,1,0,0],[0,0,0,0],[0,0,1,1],[0,0,0,0]])


#we use random symmetric matrices to build reversible matrices
def make_symmetric_matrix():
	T = np.zeros(shape=(4,4), dtype=float)
	for i in range(4):
		for j in range(i+1):
			new_entry = uniform(0,max_entry)
			T.itemset((i,j),new_entry)
	S = T + T.T - np.diag(T.diagonal())
	sum_of_all_entries = np.sum(S)
	#we normalize S so that the stationary distribution also comes out normalized
	S = (1/sum_of_all_entries)*S
	return S

#the stationary distribution is given by summing each column of the symmetric matrix
def find_stationary_distribution(S):
	stationary_distribution = np.sum(S, axis = 0)
	return stationary_distribution

#This construction guarantees reversibility of the transition matrix: M_{ij} = S_{ij}/pi_{j}
#where pi_{j} is the stationary distribution, formed by summing the jth column of S
def make_transition_matrix(S,stationary_distribution):
	M = S
	for i in range(4):
		M[:,i] *= 1/stationary_distribution.item(i)
	if embeddable == True:
		#G is a reversible generator, from which a reversibly embeddable matrix can be built
		G = M-I
		transition_matrix = LA.matrix_power(I + G*tau/n, n)
		return transition_matrix
	else:
		return M

#the energy drawn per bit from the input tape, once the beast has reached steady state
def objective(stationary_distribution,transition_matrix):
		MF1, F0MF1, dynamic_steady_state = find_dynamic_steady_state(transition_matrix)
		#reversibility demands that the stationary distribution of M is the boltzmann distribution
		#we choose the overall energy scale such that Z=1
		energies = np.array([-log(stationary_distribution[x]) for x in range(4)])
		sum_of_matrices = -I+F0MF1-MF1+F1
		sum_of_matrices_times_dynamic_steady_state = matmul(sum_of_matrices,dynamic_steady_state).tolist()
		#the factor of 1/2 is due to two bits being encountered per cycle
		objective = -(1/2)*np.dot(energies,sum_of_matrices_times_dynamic_steady_state)
		return objective

#find the steady state at the start of the full cycle given by the product MF0MF1
def find_dynamic_steady_state(transition_matrix):
	F0M = matmul(F0,transition_matrix)
	MF1 = matmul(transition_matrix,F1)
	F0MF1 = matmul(F0,MF1)
	MF0MF1 = matmul(transition_matrix,F0MF1)
	eigenvalues, eigenvectors = np.linalg.eig(MF0MF1)
	eig_norms = [np.linalg.norm(eigenvectors[:,x]) for x in range(4)]
	for i in range(4):
		column = eigenvectors[:,i]
		norm = sum(column)
		for k in range(4):
			if not norm == 0:
				column[k] = column[k]/norm
	#reversible matrices guaranteed to have real eigenvalues
	#but for numerical reasons there might be small imaginary parts
	#that should be ignored
	real_parts = [e.real for e in eigenvalues]
	#exclude matrices having eigenvalue -1 for being non-ergodic
	if all(x > -1 for x in real_parts):
		#perron-froebenius theorem guarantees a single largest eigenvalue of modulus 1
		#whose eigenvector is the steady state distributiion
		index_of_max_eigenvalue = real_parts.index((max(real_parts)))
		steady_state = eigenvectors[:, index_of_max_eigenvalue]
		steady_state_list = [prob[0] for prob in steady_state.tolist()]
		#probability distribution must be non-negative
		if any(steady_state_list[x].real < 0 for x in range(4)):
			print('error: bad steady state')
			print('transition matrix:', interaction_matrix)
			print('eigenvectors:', eigenvectors)
			print('index of max eigenvalue:', index_of_max_eigenvalue)
			print('steady state:', steady_state)
		return MF1, F0MF1, steady_state
	else:
		print('error: no steady state for this matrix')







#a model for the work-extraction device (bitbeast)
#technically it is fully defined by its transition matrix
#but we include the symmetric matrix from which it's built
#as well as the stationary distribution, for convenience
class bitBeast(object):


	def __init__(self, symmetric_matrix, stationary_distribution, transition_matrix):
		self.symmetric_matrix = symmetric_matrix
		self.stationary_distribution = stationary_distribution
		self.transition_matrix = transition_matrix
		self.objective = objective(stationary_distribution,transition_matrix)

	#pick a random entry in the symmetric matrix representation to mutate
	def pick_mutation_target(self):
		column = np.random.choice([0,1,2,3])
		row = np.random.choice([0,1,2,3])
		coordinate = (row,column)
		return coordinate

	#apply our mutation operator to the chosen coordinate
	def mutate(self,coordinate):
		symmetric_matrix = self.symmetric_matrix
		old_value = symmetric_matrix.item(coordinate)
		theta = uniform(theta_min,theta_max)
		x_intermediate = gamma*tan(theta)
		x = exp(x_intermediate)
		new_value = old_value*x
		#restrict max and min values to prevent precision issues
		if new_value > 10**(2):
			new_value = random()*10**(2)
		if new_value < 10**(-2):
			new_value = random()*10**(-2)
		return new_value

	#fix the mutation if it leads to greater fitness
	def select_fitness(self,coordinate,new_value):
		symmetric_matrix = self.symmetric_matrix
		symmetric_matrix.itemset(coordinate,new_value)
		#make sure we apply the mutation as well to the transposed entry so matrix stays symmetric
		symmetric_matrix.itemset((coordinate[1],coordinate[0]),new_value)
		sum_of_all_entries = np.sum(symmetric_matrix)
		symmetric_matrix = (1/sum_of_all_entries)*symmetric_matrix
		new_stationary_distribution = find_stationary_distribution(symmetric_matrix)
		new_transition_matrix = make_transition_matrix(symmetric_matrix,new_stationary_distribution)
		new_objective = objective(new_stationary_distribution,new_transition_matrix)
		if new_objective > self.objective:
			self.symmetric_matrix = symmetric_matrix
			self.stationary_distribution = new_stationary_distribution
			self.transition_matrix = new_transition_matrix
			self.objective = new_objective


	#mutate the transition matrix and commit the change if it leads to greater work extraction
	def adapt(self):
		coordinate = self.pick_mutation_target()
		new_value = self.mutate(coordinate)
		self.select_fitness(coordinate,new_value)

	def check_reversibility(self):
		M = self.transition_matrix
		pi = self.stationary_distribution
		for i in range(4):
			for j in range(4):
					if M.item(i,j)*pi[j] - M.item(j,i)*pi[i] > 10**(-6):
						print('--------------------')
						print('the', i, j, 'entry does not satisfy detailed balance')
						print('difference is:', M.item(i,j)*pi[j] - M.item(j,i)*pi[i])

#create starting seed for the evolutionary algorithm
def make_seed_ancestor():
	parent_symmetric_matrix = make_symmetric_matrix()
	parent_stationary_distribution = find_stationary_distribution(parent_symmetric_matrix)
	parent_transition_matrix = make_transition_matrix(parent_symmetric_matrix,parent_stationary_distribution)
	return bitBeast(parent_symmetric_matrix,parent_stationary_distribution,parent_transition_matrix)

#highest possible extracted work per bit for reversible (non-embeddable) matrices
#believed to be kT/e, and we've assumed kT=1 throughout
#adapt until we get close to this bound
def evolve(first_ancestor):
	output = []
	i = -1
	while first_ancestor.objective < 0.36:
		i +=1
		first_ancestor.adapt()
		output.append((i,first_ancestor.objective))
		if (i/10000).is_integer():
			print('------')
			print('i:', i)
			print('current objective:', first_ancestor.objective)
	return output

def save(output):
	file = open('evolution_of_fitness_4.txt','wb')
	pickle.dump(output,file)
	file.close()

def make_plot(output):
	steps = [x[0] for x in output]
	objectives = [x[1] for x in output]
	plt.plot(steps,objectives)
	plt.show()
	plt.close()


def run():
	first_ancestor = make_seed_ancestor()
	output = evolve(first_ancestor)
	save(output)
	print('max objective:', first_ancestor.objective)
	print('associated transition matrix:', first_ancestor.transition_matrix)
	print('stationary_distribution:', first_ancestor.stationary_distribution)
	print('M times stationary_distribution:', matmul(first_ancestor.transition_matrix,first_ancestor.stationary_distribution))
	first_ancestor.check_reversibility()
	make_plot(output)

run()




		 









