import matplotlib.pyplot as plt
from random import random, uniform
from math import exp
import numpy as np
from numpy import matmul
from numpy import linalg as LA
from math import log, tan
import pickle


class BitBeast:
    def __init__(self, max_entry=100, n=10 ** 9, tau=1, theta_min=-3.14 / 4, theta_max=3.14 / 4, gamma=7,
                 embeddable=False):
        # upper bound on matrix entries
        self.max_entry = max_entry
        # interaction timescale
        self.tau = tau
        # any large number, used to approximate the matrix exponential
        self.n = n
        # bounds for the mutation operator: old values multiplied by exp(gamma*tan(theta))
        # where theta is sampled uniformly from (theta_min,theta_max)
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.gamma = gamma
        # do we require generated matrices to be embeddable, or merely to be reversible?
        self.embeddable = embeddable
        self.I = np.identity(4)
        # the matrix that models feeding a 1 from the input tape to the beast
        self.F1 = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]])
        # the matrix that models feeding a 0 from the input tape to the beast
        self.F0 = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
        self.S = self.make_symmetric_matrix()
        self.stationary_distribution = self.find_stationary_distribution(self.S)
        self.transition_matrix = self.make_transition_matrix(self.S, self.stationary_distribution)
        self.objective_value = self.objective_function(self.transition_matrix, self.stationary_distribution)

    # we use random symmetric matrices to build reversible matrices
    def make_symmetric_matrix(self):
        T = np.zeros(shape=(4, 4), dtype=float)
        for i in range(4):
            for j in range(i + 1):
                new_entry = uniform(0, self.max_entry)
                T.itemset((i, j), new_entry)
        S = T + T.T - np.diag(T.diagonal())
        sum_of_all_entries = np.sum(S)
        # we normalize S so that the stationary distribution also comes out normalized
        S = (1 / sum_of_all_entries) * S
        return S

    # the stationary distribution is given by summing each column of the symmetric matrix
    @staticmethod
    def find_stationary_distribution(symmetric_matrix):
        stationary_distribution = np.sum(symmetric_matrix, axis=0)
        return stationary_distribution

    # This construction guarantees reversibility of the transition matrix: M_{ij} = S_{ij}/pi_{j}
    # where pi_{j} is the stationary distribution, formed by summing the jth column of S
    def make_transition_matrix(self, symmetric_matrix, stationary_distribution):
        M = symmetric_matrix
        I = self.I
        tau = self.tau
        n = self.n
        for i in range(4):
            M[:, i] *= 1 / stationary_distribution.item(i)
        if self.embeddable:
            # G is a reversible generator, from which a reversibly embeddable matrix can be built
            G = M - I
            transition_matrix = LA.matrix_power(I + G * tau / n, n)
            return transition_matrix
        else:
            return M

    # find the steady state at the start of the full cycle given by the product MF0MF1
    def find_dynamic_steady_state(self, transition_matrix):
        F1 = self.F1
        F0 = self.F0
        MF1 = matmul(transition_matrix, F1)
        F0MF1 = matmul(F0, MF1)
        MF0MF1 = matmul(transition_matrix, F0MF1)
        eigenvalues, eigenvectors = np.linalg.eig(MF0MF1)
        for i in range(4):
            column = eigenvectors[:, i]
            norm = sum(column)
            for k in range(4):
                if not norm == 0:
                    column[k] = column[k] / norm
        # reversible matrices guaranteed to have real eigenvalues
        # but for numerical reasons there might be small imaginary parts
        # that should be ignored
        real_parts = [e.real for e in eigenvalues]
        # exclude matrices having eigenvalue -1 for being non-ergodic
        if all(x > -1 for x in real_parts):
            # perron-froebenius theorem guarantees a single largest eigenvalue of modulus 1
            # whose eigenvector is the steady state distributiion
            index_of_max_eigenvalue = real_parts.index((max(real_parts)))
            steady_state = eigenvectors[:, index_of_max_eigenvalue]
            steady_state_list = steady_state.tolist()
            # probability distribution must be non-negative
            if any(steady_state_list[x].real < 0 for x in range(4)):
                print('error: bad steady state')
                print('full cycle transition matrix:', MF0MF1)
                print('eigenvectors:', eigenvectors)
                print('index of max eigenvalue:', index_of_max_eigenvalue)
                print('steady state:', steady_state)
            return MF1, F0MF1, steady_state
        else:
            print('error: no steady state for this matrix')

    # the energy drawn per bit from the input tape, once the beast has reached steady state
    def objective_function(self, transition_matrix, stationary_distribution):
        I = self.I
        F1 = self.F1
        MF1, F0MF1, dynamic_steady_state = self.find_dynamic_steady_state(transition_matrix)
        # reversibility demands that the stationary distribution of M is the boltzmann distribution
        # we choose the overall energy scale such that Z=1
        energies = np.array([-log(stationary_distribution[x]) for x in range(4)])
        sum_of_matrices = -I + F0MF1 - MF1 + F1
        sum_of_matrices_times_dynamic_steady_state = matmul(sum_of_matrices, dynamic_steady_state).tolist()
        # the factor of 1/2 is due to two bits being encountered per cycle
        objective = -(1 / 2) * np.dot(energies, sum_of_matrices_times_dynamic_steady_state)
        return objective


class Adaptation:

    def __init__(self):
        self.model = BitBeast()
        self.symmetric_matrix = self.model.S
        self.stationary_distribution = self.model.stationary_distribution
        self.transition_matrix = self.model.transition_matrix
        self.objective_value = self.model.objective_value
        self.output = self.evolve()

    # pick a random entry in the symmetric matrix representation to mutate
    @staticmethod
    def pick_mutation_target():
        column = np.random.choice([0, 1, 2, 3])
        row = np.random.choice([0, 1, 2, 3])
        coordinate = (row, column)
        return coordinate

    # apply our mutation operator to the chosen coordinate
    def mutate(self, coordinate):
        symmetric_matrix = self.symmetric_matrix
        old_value = symmetric_matrix.item(coordinate)
        theta = uniform(self.model.theta_min, self.model.theta_max)
        x_intermediate = self.model.gamma * tan(theta)
        x = exp(x_intermediate)
        new_value = old_value * x
        # restrict max and min values to prevent precision issues
        if new_value > 10 ** 2:
            new_value = random() * 10 ** 2
        if new_value < 10 ** (-2):
            new_value = random() * 10 ** (-2)
        return new_value

    # fix the mutation if it leads to greater fitness
    def select_fitness(self, coordinate, new_value):
        symmetric_matrix = self.symmetric_matrix
        symmetric_matrix.itemset(coordinate, new_value)
        # make sure we apply the mutation as well to the transposed entry so matrix stays symmetric
        symmetric_matrix.itemset((coordinate[1], coordinate[0]), new_value)
        sum_of_all_entries = np.sum(symmetric_matrix)
        symmetric_matrix = (1 / sum_of_all_entries) * symmetric_matrix
        new_stationary_distribution = self.model.find_stationary_distribution(symmetric_matrix)
        new_transition_matrix = self.model.make_transition_matrix(symmetric_matrix, new_stationary_distribution)
        new_objective_value = self.model.objective_function(new_transition_matrix, new_stationary_distribution)
        if new_objective_value > self.objective_value:
            self.symmetric_matrix = symmetric_matrix
            self.stationary_distribution = new_stationary_distribution
            self.transition_matrix = new_transition_matrix
            self.objective_value = new_objective_value

    # mutate the transition matrix and commit the change if it leads to greater work extraction
    def adapt(self):
        coordinate = self.pick_mutation_target()
        new_value = self.mutate(coordinate)
        self.select_fitness(coordinate, new_value)

    def check_reversibility(self):
        M = self.transition_matrix
        pi = self.stationary_distribution
        for i in range(4):
            for j in range(4):
                if M.item(i, j) * pi[j] - M.item(j, i) * pi[i] > 10 ** (-6):
                    print('--------------------')
                    print('the', i, j, 'entry does not satisfy detailed balance')
                    print('difference is:', M.item(i, j) * pi[j] - M.item(j, i) * pi[i])

    # highest possible extracted work per bit for reversible (non-embeddable) matrices
    # believed to be kT/e, and we've assumed kT=1 throughout
    # adapt until we get close to this bound
    def evolve(self):
        output = []
        i = -1
        while self.objective_value < 0.36:
            i += 1
            self.adapt()
            output.append((i, self.objective_value))
            if (i / 10000).is_integer():
                print('i:', i)
                print('current objective value:', self.objective_value)
                print('------')
        return output

    def save(self):
        file = open('evolution_of_fitness.txt', 'wb')
        pickle.dump(self.output, file)
        file.close()

    def make_plot(self):
        steps = [x[0] for x in self.output]
        objectives = [x[1] for x in self.output]
        plt.plot(steps, objectives)
        plt.show()
        plt.close()

    def report(self):
        self.save()
        print('max objective:', self.objective_value)
        print('associated transition matrix:', self.transition_matrix)
        print('stationary_distribution:', self.stationary_distribution)
        print('M times stationary_distribution:',
              matmul(self.transition_matrix, self.stationary_distribution))
        self.check_reversibility()
        self.make_plot()


evolved_beast = Adaptation()
evolved_beast.report()
