
# coding: utf-8

# In[2]:


# --- IMPORTS

from gibbs import *
import csv

file_name = "gs_maxcut.txt"
file1 = open(file_name,"r")
first_line = file1.readline()[:-1].split("\t") # first line

all_files = []
for line in file1:
    arr = line[:-1].split("\t")
    all_files += [[int(elem) for elem in arr]]

os.chdir("maxcut")



# csv_names = ['continuous.csv', 'discrete.csv', 'continuous_rbm.csv', 'discrete_rbm.csv']
csv_names = ['continuous_rbm.csv', 'discrete_rbm.csv']

for csv_name in csv_names:
    with open(csv_name, 'w') as csvfile:
        writer1 = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        result = [[],[],[],[]]
        for i in range(15):
            prob_size_max_cuts = []
            # for j in range(10):
            for j in range(1):
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
                max_cut_value = max_cut(entry_filename, 1, 10000, csv_name) #cut size
                prob_size_max_cuts += [max_cut_value/_cut] #how close it gets to cut value
            writer1.writerow([all_files[i*10][0]] + prob_size_max_cuts)
            result[0] += [all_files[i*10][0]] #prob size
            result[1] += [np.mean(prob_size_max_cuts)] #avg trials needed
            result[2] += [np.max(prob_size_max_cuts)]
            result[3] += [np.min(prob_size_max_cuts)]
    print(result)
