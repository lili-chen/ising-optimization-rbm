# returns csv with cut size for each file, after running NUM_TRIALS samples
# used for plotting the efficiency vs. problem size (see read_monte_carlo_from_csv.ipynb)

from gibbs import *
import csv

NUM_TRIALS = 10000

file_name = "gs_maxcut.txt"
file1 = open(file_name,"r")
first_line = file1.readline()[:-1].split("\t") # first line

all_files = []
for line in file1:
    arr = line[:-1].split("\t")
    all_files += [[int(elem) for elem in arr]]

os.chdir("maxcut")


csv_names = ['continuous.csv', 'discrete.csv', 'continuous_rbm.csv', 'discrete_rbm.csv']

# for all samplers
for csv_name in csv_names:
    with open(csv_name, 'w') as csvfile:
        writer1 = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result = [[],[],[],[]]
        # for all problem sizes
        for i in range(15):
            prob_size_max_cuts = []
            # for all problems of this problem size
            for j in range(10):
                # get the filename and correct cut
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
                # run Gibbs sampler on this file for NUM_TRIALS and get the cut size at the end
                max_cut_value = max_cut(entry_filename, 1, NUM_TRIALS, csv_name)
                # compare this to the actual cut value _cut
                if max_cut_value/_cut <= 1:
                    prob_size_max_cuts += [max_cut_value/_cut]
            print(prob_size_max_cuts)
            # write relevant information to csv
            writer1.writerow([all_files[i*10][0]] + prob_size_max_cuts)
