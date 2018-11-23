import numpy as np
import pandas as pd
import os, shutil, time, h5py

root_dir = './results8349/'  # Unique results directory
filepath = os.path.join(root_dir, 'file{0:03d}.csv')
hdfpath = os.path.join(root_dir, 'results.h5')

n_files = 1000
n_rows = 1000
n_cols = 2

# Clear previous results
if os.path.isdir(root_dir):
    shutil.rmtree(root_dir)
# Create new directory
os.makedirs(root_dir)

# Create fake data in many csv files
for i in range(n_files):
    results = np.random.random((n_rows, n_cols))
    np.savetxt(filepath.format(i), results, delimiter=',')

# Convert the many csv files into a single hdf file
start_time = time.time()

h5f = h5py.File(hdfpath, 'w')

data = np.zeros((n_files, n_rows, n_cols), dtype=float)

for i in range(n_files):

    X = pd.read_csv(filepath.format(i), index_col=None, header=None)
    data[i, :, :] = X.values

h5f.create_dataset('data', data=data)
h5f.close()
print('%s seconds' % (time.time() - start_time))
