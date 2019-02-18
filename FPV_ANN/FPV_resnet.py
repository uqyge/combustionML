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
from utils.customObjects import coeff_r2,SGDRScheduler

# define the labels

# labels = ['T','CH4']
# labels = ['T','CH4','O2','CO2','CO','H2O','H2','OH','PVs']
# labels = ['C2H3', 'C2H6', 'CH2', 'H2CN', 'C2H4', 'H2O2', 'C2H',
#        'CN', 'heatRelease', 'NCO', 'NNH', 'N2', 'AR', 'psi', 'CO', 'CH4',
#        'HNCO', 'CH2OH', 'HCCO', 'CH2CO', 'CH', 'mu', 'C2H2', 'C2H5', 'H2', 'T',
#        'PVs', 'O', 'O2', 'N2O', 'C', 'C3H7', 'CH2(S)', 'NH3', 'HO2', 'NO',
#        'HCO', 'NO2', 'OH', 'HCNO', 'CH3CHO', 'CH3', 'NH', 'alpha', 'CH3O',
#        'CO2', 'CH3OH', 'CH2CHO', 'CH2O', 'C3H8', 'HNO', 'NH2', 'HCN', 'H', 'N',
#        'H2O', 'HCCOH', 'HCNN']
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

input_features=['f','pv','zeta']

# define the type of scaler: MinMax or Standard
scaler = 'MinMax'

# read in the data
X, y, df, in_scaler, out_scaler = read_h5_data('./data/tables_of_fgm_psi.h5', input_features=input_features, labels = labels,i_scaler='no',o_scaler='std2')

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01)

# %%
print('set up ANN')
# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]
n_neuron = 900
branches = 1
scale = 1
batch_size = 1024*4*2
vsplit = 0.1
batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

# less then 2 res_block, there will be variance
x = res_block(x, scale, n_neuron, stage=1, block='a', bn=batch_norm,branches=branches)
x = res_block(x, scale, n_neuron, stage=1, block='b', bn=batch_norm,branches=branches)

x = Dense(100, activation='relu')(x)
# x = Dropout(0.1)(x)
predictions = Dense(dim_label, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)

model.summary()


#%%
loss_type='mse'
sgd = keras.optimizers.SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=loss_type,
              optimizer='adam',
              # optimizer=sgd,
              metrics=[coeff_r2])

# checkpoint (save the best model based validate loss)
filepath = "./tmp/weights.best.cntk.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=10)

epoch_size=X_train.shape[0]
a=0
base=2
clc=2
for i in range(7):
  a+=base*clc**(i)
print(a)
epochs,c_len = a,base
schedule = SGDRScheduler(min_lr=1e-5,max_lr=1e-4,
                         steps_per_epoch=np.ceil(epoch_size/batch_size),
                         cycle_length=c_len,lr_decay=0.8,mult_factor=2)

callbacks_list = [checkpoint,schedule]
# callbacks_list = [checkpoint]
# fit the model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list,
    shuffle=True)

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


# #%%
# model.load_weights("./tmp/weights.best.cntk.hdf5")
# # cntk.combine(model.outputs).save('mayerTest.dnn')
#
# # # %%
# # ref = df.loc[df['p'] == 40]
# # x_test = in_scaler.transform(ref[['p', 'he']])
#
# predict_val = model.predict(X_test)
#
# X_test_df = pd.DataFrame(in_scaler.inverse_transform(X_test),columns=input_features)
# y_test_df = pd.DataFrame(out_scaler.inverse_transform(y_test),columns=labels)
#
# sp='PVs'
#
# predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)
#
# # plt.figure()
# # plt.plot(X_test_df['f'], y_test_df[sp], 'r:')
# # plt.plot(X_test_df['f'], predict_df[sp], 'b-')
# # plt.show()
#
# plt.figure()
# plt.title('Error of %s ' % sp)
# plt.plot((y_test_df[sp] - predict_df[sp]) / y_test_df[sp])
# plt.title(sp)
# plt.show()
#
# plt.figure()
# plt.scatter(predict_df[sp],y_test_df[sp],s=1)
# plt.title(sp)
# plt.show()
# # %%
# a=(y_test_df[sp] - predict_df[sp]) / y_test_df[sp]
# test_data=pd.concat([X_test_df,y_test_df],axis=1)
# pred_data=pd.concat([X_test_df,predict_df],axis=1)
#
# test_data.to_hdf('sim_check.H5',key='test')
# pred_data.to_hdf('sim_check.H5',key='pred')
#
# # Save model
# sess = K.get_session()
# saver = tf.train.Saver(tf.global_variables())
# saver.save(sess, './exported/my_model')
# model.save('FPV_ANN.H5')
#
# # write the OpenFOAM ANNProperties file
# writeANNProperties(in_scaler,out_scaler,scaler)
