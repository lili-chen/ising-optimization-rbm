
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
    def get_cut_sizes_discrete(self, num_trials, W, b, step_size):
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
                cuts += [(k, self.cut_size(W, with_zeros))]

        return cuts

    def get_cut_sizes_discrete_rbm(self, num_trials, W, b, step_size):
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
                cuts += [(k, self.cut_size_rbm(W, with_zeros))]

        return cuts

    def get_cut_sizes_continuous(self, num_trials, W, b):
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
            cuts += [(total_time, self.cut_size(W, with_zeros))]

        return cuts

    def get_cut_sizes_continuous_rbm(self, num_trials, W, b):
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
            cuts += [(total_time, self.cut_size_rbm(W, with_zeros))]

        return cuts

    def cut_size_rbm(self, W, joint_state):
        num_vertices = int(len(W)/2)

        # calculate cut
        cut = 0
        for i in range(num_vertices):
            outgoing_edges = W[i]
            for j in range(len(outgoing_edges)):
                if outgoing_edges[j] != 0: # edge exists between vertex i and vertex j
                    if num_vertices+i < j and joint_state[i] != joint_state[j] and abs(i-j) != num_vertices: # on different sides of the cut
                        cut += 1
        return cut


    def cut_size(self, W, joint_state):
        num_vertices = len(W)

        # calculate cut
        cut = 0
        for i in range(num_vertices):
            outgoing_edges = W[i]
            for j in range(len(outgoing_edges)):
                if outgoing_edges[j] != 0: # edge exists between vertex i and vertex j
                    if i < j and joint_state[i] != joint_state[j]: # on different sides of the cut
                        cut += 1
                        # print(i, j, joint_state[i], joint_state[j])
        # print("---")
        return cut


# In[3]:


# --- FUNCTIONS

def max_cut(file_name, num_repetitions, num_trials):
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

    cg = Max_Cut_Approximator(num_vertices)
    b = np.zeros(num_vertices)

    result = []
    for _ in range(num_repetitions):
        dist = cg.get_distribution(num_trials, -0.5*W, b)
        best_state = np.argmax(dist)
        binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
        with_zeros = [-1] * (num_vertices - len(binary)) + binary
        # calculate Hamiltonian
        H = 0
        for i in range(num_vertices):
            for j in range(num_vertices):
                H -= 2 * -W[i][j] * with_zeros[i] * with_zeros[j] # J = -W[i][j]

        cut = cg.cut_size(W, with_zeros)

        result += [(best_state, H, cut)]
    return result

def plot_max_cut_efficiency_discrete(file_name, num_trials, step_size):
    file1 = open(file_name,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])

    W = np.zeros((num_vertices, num_vertices))

    for line in file1:
        arr = line[:-1].split("\t")
        ind1 = int(arr[0])
        ind2 = int(arr[1])
        ind3 = float(arr[2])
        W[ind1-1][ind2-1] = ind3
        W[ind2-1][ind1-1] = ind3

    cg = Max_Cut_Approximator(num_vertices)
    b = np.zeros(num_vertices)

    cuts = np.array(cg.get_cut_sizes_discrete(num_trials, -0.5*W, b, step_size))
    cuts = cuts.reshape(len(cuts), 2).T
    plt.plot(cuts[0], cuts[1])
    plt.show()

def plot_max_cut_efficiency_continuous(file_name, num_trials):
    file1 = open(file_name,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])

    W = np.zeros((num_vertices, num_vertices))

    for line in file1:
        arr = line[:-1].split("\t")
        ind1 = int(arr[0])
        ind2 = int(arr[1])
        ind3 = float(arr[2])
        W[ind1-1][ind2-1] = ind3
        W[ind2-1][ind1-1] = ind3

    cg = Max_Cut_Approximator(num_vertices)
    b = np.zeros(num_vertices)

    cuts = np.array(cg.get_cut_sizes_continuous(num_trials, -0.5*W, b))
    cuts = cuts.reshape(len(cuts), 2).T
    plt.plot(cuts[0], cuts[1])
    plt.show()

def plot_max_cut_efficiency_continuous_rbm(file_name, num_trials):
    file1 = open(file_name,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])

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
        total = np.sum(W[i])
        W[i][num_vertices+i] = -total
        W[num_vertices+i][i] = -total

    cg = Max_Cut_Approximator(2*num_vertices)
    b = np.zeros(2*num_vertices)

    cuts = np.array(cg.get_cut_sizes_continuous_rbm(num_trials, -0.5*W, b))
    cuts = cuts.reshape(len(cuts), 2).T
    plt.plot(cuts[0], cuts[1])
    plt.show()

def plot_max_cut_efficiency_discrete_rbm(file_name, num_trials, step_size):
    file1 = open(file_name,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])

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
        total = np.sum(W[i])
        W[i][num_vertices+i] = -total
        W[num_vertices+i][i] = -total

    cg = Max_Cut_Approximator(2*num_vertices)
    b = np.zeros(2*num_vertices)

    cuts = np.array(cg.get_cut_sizes_discrete_rbm(num_trials, -0.5*W, b, step_size))
    cuts = cuts.reshape(len(cuts), 2).T
    plt.plot(cuts[0], cuts[1])
    plt.show()

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
