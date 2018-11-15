import pandas as pd
import numpy as np
import os, sys

# promt input for the case
path = input('Which case? ')

headers = 20
rows = 4547099

timesteps = os.listdir(path)
timesteps = [f for f in timesteps if os.path.isdir(path) and f[0] == '0']   # checks if it is time step, first entry = 0

# the sort the time steps!
timesteps.sort()

columnsTrain = ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3','CH3OH', 'CH4',
               'CO', 'CO2', 'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2','OH', 'T', 'nut', 'Trace_gradU','mag_gradU']

# loop over the different time directories
try:
    for time in timesteps:
        print('At time: ', time)
        data = pd.DataFrame(index=range(rows))
        for col in columnsTrain:
            read_path = path + '/' + time + '/'
            data_in = pd.read_csv(read_path + col, header=headers, nrows=rows)
            # data_in.columns = list(col)
            # add the column to the table
            data[col] = data_in.values
        data.to_csv(read_path + 'dataframe_' + time + '.csv')
        print('Data frame: ' + read_path + 'dataframe_' + time + '.csv')
except KeyboardInterrupt:
    print('Stopped')