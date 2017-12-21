import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import os, sys

# promt input for the case
path = input('Which case? ')

timesteps = os.listdir(path)
timesteps = [f for f in timesteps if os.path.isdir(path) and f[0] == '0']   # checks if it is time step, first entry = 0

# the sort the time steps!
timesteps.sort()

columnsTrain= ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3','CH3OH', 'CH4',
               'CO', 'CO2', 'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2','OH', 'T']

timesteps = timesteps[0:7]

for entry, t in enumerate(timesteps):
    print('Reading in data from: ' + t)
    data = pd.read_csv(path + '/' + t + '/dataFrame_' + t + '.csv', index_col=0, low_memory=True)
    data = data[:-1]  # drops the last row with ) ) )
    print('Entry: ' + str(entry))
    # generate the training data!
    if entry < (len(timesteps) - 2):
        print('Training data')
        # generates the input array for the ANN
        if entry < len(timesteps) - 3:
            if entry == 0:
                print('Entry == 0')
                Train_arr_inp = data[columnsTrain]
            else:
                print('Entry >0')
                Train_arr_inp = Train_arr_inp.append(data[columnsTrain])
                print(Train_arr_inp.values.shape)

        # generates the input array for the ANN with shift of 1
        if entry > 0:
            print(entry)
            if entry == 1:
                Train_arr_out = data[columnsTrain]
            else:
                Train_arr_out = Train_arr_out.append(data[columnsTrain])

    # set up the test array!
    else:
        print('Entry >= (len(timesteps) - 2)')
        print('Test data\n')
        if t == timesteps[-2]:
            Test_arr_inp = data[columnsTrain]
        elif t == timesteps[-1]:
            Test_arr_out = data[columnsTrain]


def removeLowT(Train_arr_inp,Train_arr_out,Test_arr_inp,Test_arr_out):

