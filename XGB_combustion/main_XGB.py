# to play with the ANN toolbox

import matplotlib.pyplot as plt
import datetime


run prepareData.py

# this only works, if the train and test data is already in the workspace
ANN_combustion = ANN_combustion_Toolbox(Train_inp=X_Train_arr ,Train_out=y_Train_arr ,Test_inp=X_Test_arr ,Test_out=y_Test_arr)

# only predict the target values

#targets = [ 'T']

ANN_combustion.scale_data()
start = datetime.datetime.now()
ANN_combustion.gridSearchResNet( neurons = [500], layers = [5], epochs = [500], batch= [100000],loss_func = ['mse'])
end = datetime.datetime.now()

ANN_combustion.plotAccuracy(target='T')

