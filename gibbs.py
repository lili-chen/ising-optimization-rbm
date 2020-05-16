
# coding: utf-8


# --- IMPORTS

import numpy as np
import random
import matplotlib.pyplot as plt
import os


# --- CLASSES

class Gibbs:
    def __init__(self, size):
        self.size = size
        self.base_lambda = 1

    def get_distribution(self, num_trials, W, b):
        pass

    def plot_distribution(self, num_trials, W, b):
        """
        Plots the distribution from self.get_distribution as a bar graph.
        """
        plt.bar(np.arange(0, 2**self.size), self.get_distribution(num_trials, W, b))
        plt.show()

    def state_to_int(self, s):
        """
        Returns integer representation of state s (array of 1's and -1's).
        (Example: state [-1, 1, -1] corresponds to integer value 2)
        Used for creating distribution array when running Gibbs samplers.
        """
        binary = [0 if i == -1 else 1 for i in s]
        result = 0
        power_of_two = 1
        for bit in reversed(binary):
            result += power_of_two * bit
            power_of_two *= 2
        return result

class Max_Cut_Approximator(Gibbs):
    def get_cut_sizes_discrete(self, num_trials, W, b, step_size, is_rbm):
        """
        Runs discrete Gibbs sampling algorithm, calculating the size of the cut every step_size samples and adding to cuts.
        Returns cuts, and result, the probability distribution after num_trials samples were taken.
        """
        random.seed(43)
        np.random.seed(43)

        s = [random.choice([-1, 1]) for i in range(self.size)]
        result = dict()

        cuts = []
        for k in range(num_trials):
            # update the state of each neuron sequentially
            for i in range(self.size):
                summation = np.dot(W[i], s)
                s[i] = np.sign(np.tanh(summation + b[i]) + random.uniform(-1, 1))

            # update the distribution accordingly to account for the new resulting state
            if self.state_to_int(s) not in result:
                result[self.state_to_int(s)] = 0
            result[self.state_to_int(s)] += 1

            # every step_size samples (to avoid computational slowdown), calculate the size of the cut and add to cuts
            if k % step_size == 0:
                best_state = max(result, key=result.get)
                binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
                with_zeros = [-1] * (self.size - len(binary)) + binary
                cuts += [(k, self.cut_size(W, with_zeros, is_rbm))]

        # if rbm, calculate the probability distribution on the original graph
        # return cuts array (for plotting) and the probability distribution
        if is_rbm:
            return cuts, self.map_rbm_result(result, int(len(W)/2))
        return cuts, result

    def get_cut_sizes_continuous(self, num_trials, W, b, is_rbm):
        """
        Runs continuous Gibbs sampling algorithm, calculating the size of the cut as we go and adding to cuts.
        Returns cuts, and result, the probability distribution after num_trials time has passed.
        """
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

            # calculate the cut and add to cuts
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (self.size - len(binary)) + binary
            cuts += [(total_time, self.cut_size(W, with_zeros, is_rbm))]

        # if rbm, calculate the probability distribution on the original graph
        # return cuts array (for plotting) and the probability distribution
        if is_rbm:
            return cuts, self.map_rbm_result(result, int(len(W)/2))
        return cuts, result

    def get_distribution_continuous(self, num_trials, W, b):
        """
        Runs continuous Gibbs sampling algorithm and returns the probability distribution after num_trials time has passed.
        Same as get_cut_sizes_continuous, except doesn't track cut.
        """
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
        """
        Runs discrete Gibbs sampling algorithm and returns the probability distribution after num_trials time has passed.
        Same as get_cut_sizes_discrete, except doesn't track cut.
        """
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
        """
        Takes the weight matrix, array joint_state representing the state of each neuron, and returns cut size.
        """
        num_vertices = int(len(W)/2) if is_rbm else len(W)

        # calculate cut
        cut = 0
        for i in range(num_vertices):
            outgoing_edges = W[i]
            for j in range(len(outgoing_edges)):
                # edge exists between vertex i and vertex j
                if outgoing_edges[j] != 0:
                    # on different sides of the cut
                    if is_rbm * num_vertices+i < j and joint_state[i] != joint_state[j]:
                        cut += 1
        return cut

    def map_rbm_result(self, result, num_vertices):
        """
        Maps state returned by RBM back to state of original vertices (essentially cuts off second half of the state).
        """
        result_mapped = dict()

        for state in result.keys():
            binary = [0 if i == 0 else 1 for i in list(map(int, bin(state)[2:]))]
            with_zeros_all = [0] * (self.size - len(binary)) + binary
            with_zeros = with_zeros_all[:num_vertices] # just first half
            out = 0
            for bit in with_zeros:
                out = (out << 1) | bit
            result_mapped[out] = result[state]

        return result_mapped


# --- FUNCTIONS

def max_cut(file_name, num_repetitions, num_trials, csv_name):
    """
    Run the sampler according to csv_name ("continuous.csv", "discrete.csv", "continuous_rbm.csv", "discrete_rbm.csv"),
    return the final cut size num_repetitions times, each for num_trials samples.
    """
    file1 = open(file_name,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])

    for _ in range(num_repetitions):
        if csv_name == 'continuous.csv':
            # create weight matrix and bias vector
            W, b = get_W_and_b(file1, num_vertices)
            cg = Max_Cut_Approximator(num_vertices)
            result = cg.get_distribution_continuous(num_trials, -0.5*W, b)
            # using the probability distribution returned by the sampler, get the best state and calculate its cut
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (num_vertices - len(binary)) + binary
            cut = cg.cut_size(W, with_zeros, False)
        elif csv_name == 'discrete.csv':
            # create weight matrix and bias vector
            W, b = get_W_and_b(file1, num_vertices)
            dg = Max_Cut_Approximator(num_vertices)
            result = dg.get_distribution_discrete(int(num_trials/num_vertices), -0.5*W, b)
            # using the probability distribution returned by the sampler, get the best state and calculate its cut
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (num_vertices - len(binary)) + binary
            cut = dg.cut_size(W, with_zeros, False)
        elif csv_name == 'continuous_rbm.csv':
            # create weight matrix and bias vector (for RBM the weight matrix will be 2n*2n)
            W, b = get_W_and_b_rbm(file1, num_vertices, 10)
            cg = Max_Cut_Approximator(2*num_vertices)
            result = cg.get_distribution_continuous(num_trials, -0.5*W, b)
            # using the probability distribution returned by the sampler, get the best state and calculate its cut
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (2*num_vertices - len(binary)) + binary
            cut = cg.cut_size(W, with_zeros, True)
        else:
            # create weight matrix and bias vector (for RBM the weight matrix will be 2n*2n)
            W, b = get_W_and_b_rbm(file1, num_vertices, 10)
            dg = Max_Cut_Approximator(2*num_vertices)
            result = dg.get_distribution_discrete(int(num_trials/2), -0.3*W, b)
            # using the probability distribution returned by the sampler, get the best state and calculate its cut
            best_state = max(result, key=result.get)
            binary = [-1 if i == 0 else 1 for i in list(map(int, bin(best_state)[2:]))]
            with_zeros = [-1] * (2*num_vertices - len(binary)) + binary
            cut = dg.cut_size(W, with_zeros, True)
    return cut

def get_W_and_b(file1, num_vertices):
    """
    Create weight matrix W and bias vector b according file1 and num_vertices.
    """
    W = np.zeros((num_vertices, num_vertices))
    for line in file1:
        arr = line[:-1].split("\t")
        ind1 = int(arr[0])
        ind2 = int(arr[1])
        ind3 = float(arr[2])
        # add undirected edges for each row in the file, using ind1, ind2, ind3
        W[ind1-1][ind2-1] = ind3
        W[ind2-1][ind1-1] = ind3
    b = np.zeros(num_vertices)
    return W, b

def get_W_and_b_rbm(file1, num_vertices, coupling):
    """
    Create weight matrix W and bias vector b according file1 and num_vertices, using constant coupling parameter.
    """
    W = np.zeros((2*num_vertices, 2*num_vertices))

    for line in file1:
        arr = line[:-1].split("\t")
        ind1 = int(arr[0])
        ind2 = int(arr[1])
        ind3 = float(arr[2])
        # since RBM graph is bipartite, add two edges for each row in file (undirected, so four total)
        W[ind1-1][num_vertices+ind2-1] = ind3
        W[num_vertices+ind1-1][ind2-1] = ind3
        W[num_vertices+ind2-1][ind1-1] = ind3
        W[ind2-1][num_vertices+ind1-1] = ind3

    # add edge between hidden and visible copies of vertex, with weight specified by coupling parameter
    for i in range(num_vertices):
        W[i][num_vertices+i] = -coupling
        W[num_vertices+i][i] = -coupling

    b = np.zeros(2*num_vertices)
    return W, b

def plot_cuts(cuts, _cut, file_name):
    """
    Plot the array cuts (returned by algorithm) and the correct max cut _cut (given by gs_maxcut.txt).
    """
    cuts = cuts.reshape(len(cuts), 2).T
    plt.plot(cuts[0], [_cut]*len(cuts[0]), label="Theoretical Max Cut")
    plt.plot(cuts[0], cuts[1], label="Algorithm Max Cut")
    plt.xlabel("Trial Number")
    plt.ylabel("Cut Size")
    plt.title(file_name)
    plt.legend()
    plt.show()

def plot_max_cut_efficiency_discrete(i, j, all_files, num_trials, step_size):
    """
    Plot the distribution resulting from discrete sampler, and return the probability distribution.
    Inputs i and j used for the problem size and id, respectively.
    (Example: i=1, j=2 corresponds to file N020-id01.txt)
    """
    # read in the relevant information and create W and b accordingly
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b(file1, num_vertices)
    approximator = Max_Cut_Approximator(num_vertices)
    cuts, result = approximator.get_cut_sizes_discrete(num_trials, -0.5*W, b, step_size, False)
    # plot cuts (returned by algorithm) and _cut (correct cut)
    cuts = np.array(cuts).T
    cuts[0] = cuts[0]*num_vertices
    cuts = cuts.T
    plot_cuts(cuts, _cut, filename)
    return result

def plot_max_cut_efficiency_continuous(i, j, all_files, num_trials):
    """
    Plot the distribution resulting from continuous sampler, and return the probability distribution.
    Inputs i and j used for the problem size and id, respectively.
    (Example: i=1, j=2 corresponds to file N020-id01.txt)
    """
    # read in the relevant information and create W and b accordingly
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b(file1, num_vertices)
    approximator = Max_Cut_Approximator(num_vertices)
    cuts, result = approximator.get_cut_sizes_continuous(num_trials, -0.5*W, b, False)
    # plot cuts (returned by algorithm) and _cut (correct cut)
    plot_cuts(np.array(cuts), _cut, filename)
    return result

def plot_max_cut_efficiency_continuous_rbm(i, j, all_files, num_trials, coupling):
    """
    Plot the distribution resulting from continuous RBM sampler, and return the probability distribution.
    Inputs i and j used for the problem size and id, respectively.
    (Example: i=1, j=2 corresponds to file N020-id01.txt)
    """
    # read in the relevant information and create W and b accordingly
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b_rbm(file1, num_vertices, coupling)
    approximator = Max_Cut_Approximator(2*num_vertices)
    cuts, result = approximator.get_cut_sizes_continuous(num_trials, -0.5*W, b, True)
    # plot cuts (returned by algorithm) and _cut (correct cut)
    plot_cuts(np.array(cuts), _cut, filename)
    return result

def plot_max_cut_efficiency_discrete_rbm(i, j, all_files, num_trials, step_size, coupling):
    """
    Plot the distribution resulting from continuous RBM sampler, and return the probability distribution.
    Inputs i and j used for the problem size and id, respectively.
    (Example: i=1, j=2 corresponds to file N020-id01.txt)
    """
    # read in the relevant information and create W and b accordingly
    filename, _cut = get_filename_and_cut(i, j, all_files)
    file1 = open(filename,"r")
    arr = file1.readline()[:-1].split("\t") # first line
    num_vertices = int(arr[0])
    W, b = get_W_and_b_rbm(file1, num_vertices, coupling)
    approximator = Max_Cut_Approximator(2*num_vertices)
    cuts, result = approximator.get_cut_sizes_discrete(num_trials, -0.1*W, b, step_size, True)
    # plot cuts (returned by algorithm) and _cut (correct cut)
    cuts = np.array(cuts).T
    cuts[0] = cuts[0]*2
    cuts = cuts.T
    plot_cuts(cuts, _cut, filename)
    return result

def get_filename_and_cut(i, j, all_files):
    """
    Get the corresponding filename and correct cut size using indices i, j.
    Inputs i and j used for the problem size and id, respectively.
    (Example: i=1, j=2 corresponds to file N020-id01.txt)
    """
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

def get_target_distribution(i, j, all_files):
    """
    Calculates theoretical distribution (not using any Gibbs sampler).
    Unpractical for large graphs, but useful sanity check for small graphs.
    """
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


# https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
