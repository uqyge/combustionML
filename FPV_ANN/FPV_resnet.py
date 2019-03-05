import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from utils.resBlock import res_block
from utils.data_reader import read_h5_data
from utils.writeANNProperties import writeANNProperties
from utils.customObjects import coeff_r2, SGDRScheduler


# define the labels
labels = []

with open('GRI_13', 'r') as f:
    species = f.readlines()
    for line in species:
        # remove linebreak which is the last character of the string
        current_place = line[:-1]
        # add item to the list
        labels.append(current_place)

# append other fields: heatrelease,  T, PVs
# labels.append('heatRelease')
labels.append('T')
labels.append('PVs')
# labels.remove('H')
# labels.remove('CH2O')
# labels.remove('HO2')
# labels.append('H')

# tabulate psi, mu, alpha
# labels.append('psi')
# labels.append('mu')
# labels.append('alpha')

# labels.remove('AR')
# labels.remove('N2')

input_features = ['f', 'zeta', 'pv']

# define the type of scaler: MinMax or Standard


# read in the data
X, y, df, in_scaler, out_scaler = read_h5_data('./data/tables_of_fgm_psi_n2fix.h5',
                                               input_features=input_features,
                                               labels=labels,
                                               i_scaler='no', o_scaler='bc')

# write the OpenFOAM ANNProperties file
scaler = 'Standard'
# writeANNProperties(in_scaler,out_scaler,scaler)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

# %%
print('set up ANN')
# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]
n_neuron = 100
branches = 3
scale = 3
batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,), name='input_1')

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

# less then 2 res_block, there will be variance
x = res_block(x, scale, n_neuron, stage=1, block='a', bn=batch_norm, branches=branches)
x = res_block(x, scale, n_neuron, stage=1, block='b', bn=batch_norm, branches=branches)
# x = res_block(x, scale, n_neuron, stage=1, block='c', bn=batch_norm, branches=branches)

x = Dense(100, activation='relu')(x)
# x = Dropout(0.1)(x)
predictions = Dense(dim_label, activation='linear', name='output_1')(x)

model = Model(inputs=inputs, outputs=predictions)

model.summary()

# %%
vsplit = 0.1
batch_size = 1024 * 8

# checkpoint (save the best model based validate loss)
filepath = "./tmp/weights.best.cntk.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=10)

epoch_size = X_train.shape[0]

a = 0
base = 2
clc = 2
for i in range(9):
    a += base * clc ** (i)
epochs, c_len = a, base

schedule = SGDRScheduler(min_lr=1e-6, max_lr=1e-4,
                         steps_per_epoch=np.ceil(epoch_size / batch_size),
                         cycle_length=c_len, lr_decay=0.8, mult_factor=clc)

callbacks_list = [checkpoint,schedule]
# callbacks_list = [checkpoint]

loss_type = 'mse'
for i in range(1):
    # fit the model

    model.compile(loss=loss_type,
                  optimizer='adam',
                  metrics=[coeff_r2])
    model.load_weights("./tmp/weights.best.cntk.hdf5")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=vsplit,
        verbose=2,
        callbacks=callbacks_list,
        shuffle=False)

# loss
fig = plt.figure()
plt.semilogy(history.history['loss'])
if vsplit:
    plt.semilogy(history.history['val_loss'])
plt.title(loss_type)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
model.save('./tmp/calc_100_3_3_100_cbrt.h5')

#%%
n_res = 501
sp='PVs'
for i in range(11):
    # pv_level = 0.98+i*0.002
    pv_level = i /10
    f_1 = np.linspace(0, 1, n_res)
    z_1 = np.zeros(n_res)
    pv_1 = np.ones(n_res) * pv_level
    case_1 = np.vstack((f_1, z_1, pv_1))
    # case_1 = np.vstack((pv_1,z_1,f_1))

    case_1 = case_1.T
    out = out_scaler.inverse_transform(model.predict(in_scaler.transform(case_1)))
    out = pd.DataFrame(out, columns=labels)
    table_val=df[(df.pv==pv_level) & (df.zeta==0)][sp]

    fig = plt.figure()
    plt.xlim([0,0.2])
    plt.plot(f_1,out[sp],'k')
    plt.plot(f_1,table_val,'rd')
    plt.title(pv_level)
    plt.show()

#%%
n_res = 501
for sp in labels:
    f_level = 0.18
    f_1 = np.ones(n_res) * f_level
    z_1 = np.zeros(n_res)
    pv_1 = np.linspace(0,1,n_res)
    case_1 = np.vstack((f_1, z_1, pv_1))
    # case_1 = np.vstack((pv_1,z_1,f_1))

    case_1 = case_1.T
    out = out_scaler.inverse_transform(model.predict(in_scaler.transform(case_1)))
    out = pd.DataFrame(out, columns=labels)
    table_val=df[(df.f==f_level) & (df.zeta==0)][sp]

    fig = plt.figure()
    plt.plot(pv_1,out[sp],'k')
    plt.plot(pv_1,table_val,'rd',ms=1)
    plt.title(sp+':'+str(f_level)+'_max_'+str(df[sp].max()))
    plt.show()

#%%
# sp='CH4'#NH3,H2
# f_level = 0.044
# f_1 = np.ones(n_res) * f_level
# z_1 = np.zeros(n_res)
# pv_1 = np.linspace(0,1,n_res)
# case_1 = np.vstack((f_1, z_1, pv_1))
# # case_1 = np.vstack((pv_1,z_1,f_1))
#
# case_1 = case_1.T
# out = out_scaler.inverse_transform(model.predict(in_scaler.transform(case_1)))
# out = pd.DataFrame(out, columns=labels)
# table_val=df[(df.f==f_level) & (df.zeta==0)][sp]
#
# fig = plt.figure()
# plt.plot(pv_1,np.cbrt(out[sp]),'k')
# plt.plot(pv_1,np.cbrt(table_val),'rd',ms=1)
# plt.title(sp+':'+str(f_level))
# plt.show()