import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

from resBlock import res_block
from data_reader import read_h5_data
#from writeANNProperties import writeANNProperties


# define the labels

# labels = ['T','CH4']
labels = ['T','CH4','O2','CO2','CO','H2O','H2','OH','PVs']
# labels = ['C2H3', 'C2H6', 'CH2', 'H2CN', 'C2H4', 'H2O2', 'C2H',
#        'CN', 'heatRelease', 'NCO', 'NNH', 'N2', 'AR', 'psi', 'CO', 'CH4',
#        'HNCO', 'CH2OH', 'HCCO', 'CH2CO', 'CH', 'mu', 'C2H2', 'C2H5', 'H2', 'T',
#        'PVs', 'O', 'O2', 'N2O', 'C', 'C3H7', 'CH2(S)', 'NH3', 'HO2', 'NO',
#        'HCO', 'NO2', 'OH', 'HCNO', 'CH3CHO', 'CH3', 'NH', 'alpha', 'CH3O',
#        'CO2', 'CH3OH', 'CH2CHO', 'CH2O', 'C3H8', 'HNO', 'NH2', 'HCN', 'H', 'N',
#        'H2O', 'HCCOH', 'HCNN']


input_features=['f','pv','zeta']

# read in the data
X, y, df, in_scaler, out_scaler = read_h5_data('./data/tables_of_fgm.h5',input_features=input_features, labels = labels)
