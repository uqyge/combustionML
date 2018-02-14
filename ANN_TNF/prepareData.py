import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import os, sys

# promt input for the case
path = input('Which case? ')

timesteps_all = os.listdir(path)
timesteps_all = [f for f in timesteps_all if os.path.isdir(path) and f[0] == '0']   # checks if it is time step, first entry = 0

# the sort the time steps!
timesteps_all.sort()

columnsTrain= ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3','CH3OH', 'CH4',
               'CO', 'CO2', 'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2','OH', 'T','nut', 'Trace_gradU','mag_gradU']

timesteps = timesteps_all[0:7]

for entry, t in enumerate(timesteps):
    print('Reading in data from: ' + t)
    data = pd.read_csv(path + '/' + t + '/dataframe_' + t + '.csv', index_col=0, low_memory=True)
    data = data[:-1]  # drops the last row with ) ) )
    print('Entry: ' + str(entry))
    # generate the training data!
    if entry < (len(timesteps) - 2):
        print('Training data')
        # generates the input array for the ANN
        if entry < len(timesteps) - 3:
            if entry == 0:
                print('Entry == 0')
                X_Train_arr = data[columnsTrain]
            else:
                print('Entry >0')
                X_Train_arr = X_Train_arr.append(data[columnsTrain])
                print(X_Train_arr.values.shape)

        # generates the input array for the ANN with shift of 1
        if entry > 0:
            print(entry)
            if entry == 1:
                y_Train_arr = data[columnsTrain]
            else:
                y_Train_arr = y_Train_arr.append(data[columnsTrain])

    # set up the test array!
    else:
        print('Entry >= (len(timesteps) - 2)')
        print('Test data\n')
        if t == timesteps[-2]:
            X_Test_arr = data[columnsTrain]
        elif t == timesteps[-1]:
            y_Test_arr = data[columnsTrain]


#def removeLowT(X_Train_arr,y_Train_arr,X_Test_arr,Test_arr_out):