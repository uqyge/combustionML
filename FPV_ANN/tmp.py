import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

from utils.resBlock import res_block
from utils.data_reader import read_hdf_data, read_hdf_data_psi
from utils.writeANNProperties import writeANNProperties
from keras import backend as K

import ast

##########################
# Parameters
n_neuron = 100 #400
branches = 3
scale = 3
batch_size = 1024*10
epochs = 2000
vsplit = 0.1
batch_norm = False

# define the type of scaler: MinMax or Standard
scaler = 'MinMax' # 'Standard'

##########################

labels = []

with open('GRI_species_order_reduced', 'r') as f:
    species = f.readlines()
    for line in species:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        labels.append(current_place)

# append other fields: heatrelease,  T, PVs
#labels.append('heatRelease')
labels.append('T')
labels.append('PVs')

# tabulate psi, mu, alpha
labels.append('psi')
labels.append('mu')
labels.append('alpha')

# DO NOT CHANGE THIS ORDER!!
input_features=['f','zeta','pv']


# read in the data
X, y, df, in_scaler, out_scaler = read_hdf_data_psi('./data/tables_of_fgm.h5',key='of_tables',
                                                in_labels=input_features, labels = labels,scaler=scaler)