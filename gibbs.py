
# coding: utf-8

# In[1]:


# --- IMPORTS

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

import cProfile


# In[2]:


# --- CLASSES

class Gibbs:
    def __init__(self, size):
        self.size = size
        self.base_lambda = 1

    def get_distribution(self, num_trials, W, b):
        pass

    def plot_distribution(self, num_trials, W, b):
        plt.bar(np.arange(0, 2**self.size), self.get_distribution(num_trials, W, b))
        plt.show()

    def state_to_int(self, s):
        binary = [0 if i == -1 else 1 for i in s]
        result = 0
        power_of_two = 1
        for bit in reversed(binary):
            result += power_of_two * bit
            power_of_two *= 2
        return result

class Max_Cut_Approximator(Gibbs):
    def get_cut_sizes_discrete(self, num_trials, W, b, step_size, is_rbm):
        random.seed(43)
        np.random.seed(43)

        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = dict()

        cuts = []
        for k in range(num_trials):
            for i in range(self.size):
                summation = np.dot(W[i], s)
                s[i] = np.sign(np.tanh(summation + b[i]) + random.uniform(-1, 1))

            if self.state_to_int(s) not in result:
                result[self.state_to_int(s)] = 0
            result[self.state_to_int(s)] += 1

            if k % step_size == 0:
                best_state = max(result, key=result.get)
                binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
                with_zeros = [-1] * (self.size - len(binary)) + binary
                cuts += [(k, self.cut_size(W, with_zeros, is_rbm))]

        if is_rbm:
            return cuts, self.map_rbm_result(result, int(len(W)/2))
        return cuts, result

    def get_cut_sizes_continuous(self, num_trials, W, b, is_rbm):
        random.seed(42)
        np.random.seed(42)

        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = dict()
        lambdas = [self.base_lambda] * self.size

        cuts = []
        total_time = 0
        while total_time < num_trials:
            # competing exponentials
            times = [-np.log(1-np.random.uniform(0,1))/lambdas[i] for i in range(self.size)]

            # update the s[i] of the exponential that hit first
            argmin_time = np.argmin(times)
            if self.state_to_int(s) not in result:
                result[self.state_to_int(s)] = 0
            result[self.state_to_int(s)] += times[argmin_time]
            s[argmin_time] = -s[argmin_time]
            total_time += argmin_time

            # update summation, sigmoid, and lambdas
            sigmoids = np.multiply(np.add(np.tanh(np.add(np.matmul(s, W), b)), 1), 0.5)
            lambdas = [(1-sigmoids[i])*self.base_lambda if s[i] == 1 else sigmoids[i]*self.base_lambda for i in range(self.size)]

            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (self.size - len(binary)) + binary
            cuts += [(total_time, self.cut_size(W, with_zeros, is_rbm))]

        if is_rbm:
            return cuts, self.map_rbm_result(result, int(len(W)/2))
        return cuts, result

    def get_distribution_continuous(self, num_trials, W, b):
        random.seed(42)
        np.random.seed(42)

        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = dict()
        lambdas = [self.base_lambda] * self.size

        cuts = []
        total_time = 0
        while total_time < num_trials:
            # competing exponentials
            times = [-np.log(1-np.random.uniform(0,1))/lambdas[i] for i in range(self.size)]

            # update the s[i] of the exponential that hit first
            argmin_time = np.argmin(times)
            if self.state_to_int(s) not in result:
                result[self.state_to_int(s)] = 0
            result[self.state_to_int(s)] += times[argmin_time]
            s[argmin_time] = -s[argmin_time]
            total_time += argmin_time

            # update summation, sigmoid, and lambdas
            sigmoids = np.multiply(np.add(np.tanh(np.add(np.matmul(s, W), b)), 1), 0.5)
            lambdas = [(1-sigmoids[i])*self.base_lambda if s[i] == 1 else sigmoids[i]*self.base_lambda for i in range(self.size)]

        return result

    def get_distribution_discrete(self, num_trials, W, b):
        random.seed(43)
        np.random.seed(43)

        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = dict()

        cuts = []
        for k in range(num_trials):
            for i in range(self.size):
                summation = np.dot(W[i], s)
                s[i] = np.sign(np.tanh(summation + b[i]) + random.uniform(-1, 1))

            if self.state_to_int(s) not in result:
                result[self.state_to_int(s)] = 0
            result[self.state_to_int(s)] += 1

        return result


    def cut_size(self, W, joint_state, is_rbm):
        num_vertices = int(len(W)/2) if is_rbm else len(W)

        # calculate cut
        cut = 0
        for i in range(num_vertices):
            outgoing_edges = W[i]
            for j in range(len(outgoing_edges)):
                if outgoing_edges[j] != 0: # edge exists between vertex i and vertex j
                    if is_rbm*num_vertices+i < j and joint_state[i] != joint_state[j]: # on different sides of the cut
                        cut += 1
        return cut

    def map_rbm_result(self, result, num_vertices):
        result_mapped = dict()

        for state in result.keys():
            binary = [0 if i == 0 else 1 for i in list(map(int, bin(state)[2:]))]
            with_zeros1 = [0] * (self.size - len(binary)) + binary
            with_zeros = with_zeros1[:num_vertices] # just first half
            out = 0
            for bit in with_zeros:
                out = (out << 1) | bit # https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
            result_mapped[out] = result[state]

        return result_mapped



# In[3]:


# --- FUNCTIONS

def max_cut(file_name, num_repetitions, num_trials, csv_name):
    file1 = open(file_name,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])

    for _ in range(num_repetitions):
        if csv_name == 'continuous.csv':
            W, b = get_W_and_b(file1, num_vertices)
            cg = Max_Cut_Approximator(num_vertices)
            result = cg.get_distribution_continuous(num_trials, -0.5*W, b)
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (num_vertices - len(binary)) + binary
            cut = cg.cut_size(W, with_zeros, False)
        elif csv_name == 'discrete.csv':
            W, b = get_W_and_b(file1, num_vertices)
            cg = Max_Cut_Approximator(num_vertices)
            result = cg.get_distribution_discrete(num_trials, -0.5*W, b)
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (num_vertices - len(binary)) + binary
            cut = cg.cut_size(W, with_zeros, False)
        elif csv_name == 'continuous_rbm.csv':
            W, b = get_W_and_b_rbm(file1, num_vertices, 10)
            cg = Max_Cut_Approximator(2*num_vertices)
            result = cg.get_distribution_continuous(num_trials, -0.5*W, b)
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (2*num_vertices - len(binary)) + binary
            cut = cg.cut_size(W, with_zeros, True)
        else:
            W, b = get_W_and_b_rbm(file1, num_vertices, 10)
            cg = Max_Cut_Approximator(2*num_vertices)
            result = cg.get_distribution_discrete(num_trials, -0.5*W, b)
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (2*num_vertices - len(binary)) + binary
            cut = cg.cut_size(W, with_zeros, True)

        # calculate Hamiltonian
        # H = 0
        # for i in range(num_vertices):
        #     for j in range(num_vertices):
        #         H -= 2 * -W[i][j] * with_zeros[i] * with_zeros[j] # J = -W[i][j]

        # result += [(best_state, H, cut)]
    return cut

def get_W_and_b(file1, num_vertices):
    W = np.zeros((num_vertices, num_vertices))
    for line in file1:
        arr = line[:-1].split("\t")
        ind1 = int(arr[0])
        ind2 = int(arr[1])
        ind3 = float(arr[2])
        W[ind1-1][ind2-1] = ind3
        W[ind2-1][ind1-1] = ind3
    b = np.zeros(num_vertices)
    return W, b

def get_W_and_b_rbm(file1, num_vertices, coupling):
    W = np.zeros((2*num_vertices, 2*num_vertices))

    for line in file1:
        arr = line[:-1].split("\t")
        ind1 = int(arr[0])
        ind2 = int(arr[1])
        ind3 = float(arr[2])
        W[ind1-1][num_vertices+ind2-1] = ind3
        W[num_vertices+ind1-1][ind2-1] = ind3

        W[num_vertices+ind2-1][ind1-1] = ind3
        W[ind2-1][num_vertices+ind1-1] = ind3

    for i in range(num_vertices):
        # total = np.sum(W[i])
        # W[i][num_vertices+i] = -coupling*total
        # W[num_vertices+i][i] = -coupling*total
        W[i][num_vertices+i] = -coupling
        W[num_vertices+i][i] = -coupling

    b = np.zeros(2*num_vertices)
    return W, b

def plot_cuts(cuts, _cut, file_name):
    cuts = cuts.reshape(len(cuts), 2).T
    plt.plot(cuts[0], [_cut]*len(cuts[0]), label="Theoretical Max Cut")
    plt.plot(cuts[0], cuts[1], label="Algorithm Max Cut")
    plt.xlabel("Trial Number")
    plt.ylabel("Cut Size")
    plt.title(file_name)
    plt.legend()
    plt.show()

def plot_max_cut_efficiency_discrete(i, j, all_files, num_trials, step_size):
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b(file1, num_vertices)
    approximator = Max_Cut_Approximator(num_vertices)
    cuts, result = approximator.get_cut_sizes_discrete(num_trials, -0.5*W, b, step_size, False)
    plot_cuts(np.array(cuts), _cut, filename)
    return result

def plot_max_cut_efficiency_continuous(i, j, all_files, num_trials):
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b(file1, num_vertices)
    approximator = Max_Cut_Approximator(num_vertices)
    cuts, result = approximator.get_cut_sizes_continuous(num_trials, -0.5*W, b, False)
    plot_cuts(np.array(cuts), _cut, filename)
    return result

def plot_max_cut_efficiency_continuous_rbm(i, j, all_files, num_trials, coupling):
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b_rbm(file1, num_vertices, coupling)
    approximator = Max_Cut_Approximator(2*num_vertices)
    cuts, result = approximator.get_cut_sizes_continuous(num_trials, -0.5*W, b, True)
    plot_cuts(np.array(cuts), _cut, filename)
    return result

def plot_max_cut_efficiency_discrete_rbm(i, j, all_files, num_trials, step_size, coupling):
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b_rbm(file1, num_vertices, coupling)
    approximator = Max_Cut_Approximator(2*num_vertices)
    cuts, result = approximator.get_cut_sizes_discrete(num_trials, -0.1*W, b, step_size, True)
    plot_cuts(np.array(cuts), _cut, filename)
    return result

def get_filename_and_cut(i, j, all_files):
    num_zeros = int(all_files[i*10+j][0]<100)
    entry = all_files[i*10+j]
    _n = entry[0]
    _id = entry[1]
    _ed = entry[2]
    _cut = entry[3]
    _H = entry[4]
    _soln = entry[5]
    entry_filename = "maxcut/N" + (num_zeros * "0") + str(_n) + "-id0" + str(_id) + ".txt"
    return entry_filename, _cut

def calculate_distribution(file_name):
    file1 = open(file_name,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])

    W = np.zeros((num_vertices, num_vertices))

    for line in file1:
        arr = line[:-1].split("\t")
        ind1 = int(arr[0])
        ind2 = int(arr[1])
        ind3 = int(arr[2])
        W[ind1-1][ind2-1] = ind3

    result = []
    for i in range(2**num_vertices):
        binary = [-1 if i == 0 else 1 for i in list(map(int, bin(i)[2:]))]
        with_zeros = [-1] * (num_vertices - len(binary)) + binary
        numerator = np.exp(np.dot(np.dot(with_zeros, W),with_zeros).T)
        result += [numerator]
    denominator = sum(result)
    return result/denominator

def get_target_distribution(i, j, all_files):
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b(file1, num_vertices)
    hamiltonians = []

    for state in range(int(2**num_vertices)):
        binary = [-1 if i == 0 else 1 for i in list(map(int, bin(state)[2:]))]
        with_zeros = [-1] * (num_vertices - len(binary)) + binary

        H = 0
        for i in range(len(W)):
            for j in range(len(W)):
                H -= -2 * W[i][j] * with_zeros[i] * with_zeros[j] # J = -W[i][j]
        hamiltonians += [-H]

    return np.exp(hamiltonians)/np.sum(np.exp(hamiltonians))
