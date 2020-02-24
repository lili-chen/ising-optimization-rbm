
# coding: utf-8

# In[1]:


# --- IMPORTS

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()


# In[5]:


# --- CLASSES

class Gibbs:
    def __init__(self, size):
        self.size = size
        self.base_lambda = 1/size

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

class Discrete_Gibbs(Gibbs):
    def get_distribution(self, num_trials, W, b):
        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = np.zeros(2**self.size)
        for k in range(num_trials):
            for i in range(self.size):
                summation = np.dot(W[i], s)
                s[i] = np.sign(np.tanh(summation + b[i]) + random.uniform(-1, 1))
                result[self.state_to_int(s)] += 1
        return result

class Continuous_Gibbs(Gibbs):
    def get_distribution(self, num_trials, W, b):
        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = np.zeros(2**self.size)
        lambdas = [self.base_lambda] * self.size
        for k in range(num_trials):
            # competing exponentials
            times = [-np.log(1-np.random.uniform(0,1))/lambdas[i] for i in range(self.size)]

            # update the s[i] of the exponential that hit first
            argmin_time = np.argmin(times)
            result[self.state_to_int(s)] += times[argmin_time]
            s[argmin_time] = -s[argmin_time]

            # update summation, sigmoid, and lambdas
            sigmoids = np.multiply(np.add(np.tanh(np.add(np.matmul(s, W), b)), 1), 0.5)
            lambdas = [(1-sigmoids[i])*self.base_lambda if s[i] == 1 else sigmoids[i]*self.base_lambda for i in range(self.size)]
        return result

class Max_Cut_Approximator(Continuous_Gibbs):
    def get_cut_sizes(self, num_trials, W, b):
        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = np.zeros(2**self.size)
        lambdas = [self.base_lambda] * self.size

        cuts = []
        for k in range(num_trials):
            # competing exponentials
            times = [-np.log(1-np.random.uniform(0,1))/lambdas[i] for i in range(self.size)]

            # update the s[i] of the exponential that hit first
            argmin_time = np.argmin(times)
            result[self.state_to_int(s)] += times[argmin_time]
            s[argmin_time] = -s[argmin_time]

            # update summation, sigmoid, and lambdas
            sigmoids = np.multiply(np.add(np.tanh(np.add(np.matmul(s, W), b)), 1), 0.5)
            lambdas = [(1-sigmoids[i])*self.base_lambda if s[i] == 1 else sigmoids[i]*self.base_lambda for i in range(self.size)]

            best_state = np.argmax(result)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (self.size - len(binary)) + binary
            cuts += [self.cut_size(W, with_zeros)]

        return cuts

    def run_until_correct(self, W, b, correct_cut):

        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = np.zeros(2**self.size)
        lambdas = [self.base_lambda] * self.size

        cut = 0
        num_trials = 0
        best_state = None
        while cut != correct_cut:
            # competing exponentials
            times = [-np.log(1-np.random.uniform(0,1))/lambdas[i] for i in range(self.size)]

            # update the s[i] of the exponential that hit first
            argmin_time = np.argmin(times)
            result[self.state_to_int(s)] += times[argmin_time]
            s[argmin_time] = -s[argmin_time]

            # update summation, sigmoid, and lambdas
            sigmoids = np.multiply(np.add(np.tanh(np.add(np.matmul(s, W), b)), 1), 0.5)
            lambdas = [(1-sigmoids[i])*self.base_lambda if s[i] == 1 else sigmoids[i]*self.base_lambda for i in range(self.size)]

            if num_trials % 100000 == 0:
                best_state = np.argmax(result)
                binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
                with_zeros = [-1] * (self.size - len(binary)) + binary

                cut = self.cut_size(W, with_zeros)

                print(num_trials, cut, correct_cut)
                print(time.time() - start_time)
                num_trials += 1
            else:
                cut = 0
                num_trials += 1

            if num_trials > 100000:
                return 100000
        return num_trials

    def cut_size(self, W, joint_state):
        num_vertices = len(W)

        # calculate cut
        cut = 0
        for i in range(num_vertices):
            outgoing_edges = W[i]
            for j in range(len(outgoing_edges)):
                if outgoing_edges[j] != 0: # edge exists between vertex i and vertex j
                    if joint_state[i] != joint_state[j]: # on different sides of the cut
                        cut += 1
        return cut


# In[6]:


# FUNCTIONS

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
        dist = cg.get_distribution(num_trials, -3*W, b)
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

def plot_max_cut_efficiency(file_name, num_trials):
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

    cuts = cg.get_cut_sizes(num_trials, -3*W, b)

    plt.plot(cuts)
    plt.show()

def cut_trials_until_correct(file_name, correct_cut):
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

    trials_required = cg.run_until_correct(-0.5*W, 0.5*b, correct_cut) # changed mixing rate

    return trials_required

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



# In[4]:


# RUNTIME


# os.chdir("../")

file_name = "gs_maxcut.txt"
file1 = open(file_name,"r")
first_line = file1.readline()[:-1].split("\t") # first line

all_files = []
for line in file1:
    arr = line[:-1].split("\t")
    all_files += [[int(elem) for elem in arr]]

os.chdir("maxcut")

# for i in range(int(len(all_files) / 10)):

result = [[],[]]
for i in range(2, 3):
    prob_size_avg_trials = []
    for j in range(10):
        num_zeros = int(all_files[i*10+j][0]<100)
        entry = all_files[i*10+j]
        _n = entry[0]
        _id = entry[1]
        _ed = entry[2]
        _cut = entry[3]
        _H = entry[4]
        _soln = entry[5]
        entry_filename = "N" + (num_zeros * "0") + str(_n) + "-id0" + str(_id) + ".txt"
        print(entry_filename)
        avg_trials_needed = cut_trials_until_correct(entry_filename, _cut) # change to 10
        prob_size_avg_trials += [avg_trials_needed]
        print(avg_trials_needed)
    print(prob_size_avg_trials)
    result[0] += [all_files[i*10][0]] #prob size
    largest = np.max(prob_size_avg_trials)
    prob_size_avg_trials.remove(largest)
    largest = np.max(prob_size_avg_trials)
    prob_size_avg_trials.remove(largest)
    result[1] += [np.mean(prob_size_avg_trials)] #avg trials needed

print(result)
plt.plot(result[0], result[1])
plt.show()
