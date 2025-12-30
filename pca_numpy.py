import csv
import numpy as np

data_array = np.loadtxt('iris.csv', delimiter=',', dtype=float, skiprows=1)
print(data_array)