import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
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


# define the target values for y
targets = ['CO2','H2O','T']

# now creat X and y Scaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_data = X_Test_arr.copy()
X_data = X_data.append(X_Train_arr)

y_data = y_Test_arr.copy()
y_data = y_data[targets]
y_data = y_data.append(y_Train_arr[targets])

scaler_X.fit(X_data)
scaler_y.fit(y_data)

# transform the data
X_test = scaler_X.transform(X_Test_arr)
X_train = scaler_X.transform(X_Train_arr)

y_test = scaler_y.transform(y_Test_arr[targets])
y_train = scaler_y.transform(y_Train_arr[targets])

