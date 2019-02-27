import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

from resBlock import res_block_org
from data_reader import read_hdf_data, read_hdf_data_psi
from writeANNProperties import writeANNProperties
from keras import backend as K
from keras.models import load_model

import ast

##########################
# Parameters
n_neuron = 500
branches = 3
scale = 3
batch_size = 1024*4
epochs = 2000
vsplit = 0.1
batch_norm = False

# define the type of scaler: MinMax or Standard
scaler = 'Standard' # 'Standard' 'MinMax'

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
X, y, df, in_scaler, out_scaler = read_hdf_data_psi('./tables_of_fgm.H5',key='of_tables',
                                                in_labels=input_features, labels = labels,scaler=scaler)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.01)

# %%
print('set up ANN')

# ANN parameters
dim_input = X_train.shape[1]
dim_label = y_train.shape[1]


# This returns a tensor
inputs = Input(shape=(dim_input,))#,name='input_1')

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)
#
# x = res_block(x, scale, n_neuron, stage=1, block='a', bn=batch_norm,branches=branches)
# x = res_block(x, scale, n_neuron, stage=1, block='b', bn=batch_norm,branches=branches)
# x = res_block(x, scale, n_neuron, stage=1, block='c', bn=batch_norm,branches=branches)


x = res_block_org(x, n_neuron, stage=1, block='a', bn=batch_norm)
x = res_block_org(x, n_neuron, stage=1, block='b', bn=batch_norm)
x = res_block_org(x, n_neuron, stage=1, block='c', bn=batch_norm)
#x = res_block(x, n_neuron, stage=1, block='d', bn=batch_norm)

predictions = Dense(dim_label, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# get the model summary
model.summary()

# checkpoint (save the best model based validate loss)
filepath = "./tmp/weights.best.cntk.hdf5"

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=10)

callbacks_list = [checkpoint]

# fit the model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list,
    shuffle=True)

#%%
model.load_weights("./tmp/weights.best.cntk.hdf5")
# cntk.combine(model.outputs).save('mayerTest.dnn')

# # %%
# ref = df.loc[df['p'] == 40]
# x_test = in_scaler.transform(ref[['p', 'he']])

predict_val = model.predict(X_test)

X_test_df = pd.DataFrame(in_scaler.inverse_transform(X_test),columns=input_features)
y_test_df = pd.DataFrame(out_scaler.inverse_transform(y_test),columns=labels)

sp='PVs'

# loss
fig = plt.figure()
plt.semilogy(history.history['loss'])
if vsplit:
    plt.semilogy(history.history['val_loss'])
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('./exported/Loss_%s_%s_%i.eps' % (sp,scaler,n_neuron),format='eps')
plt.show(block=False)

predict_df = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=labels)

plt.figure()
plt.title('Error of %s ' % sp)
plt.plot((y_test_df[sp] - predict_df[sp]) / y_test_df[sp])
plt.title(sp)
plt.savefig('./exported/Error_%s_%s_%i.eps' % (sp,scaler,n_neuron),format='eps')
plt.show(block=False)

plt.figure()
plt.scatter(predict_df[sp],y_test_df[sp],s=1)
plt.title('R2 for '+sp)
plt.savefig('./exported/R2_%s_%s_%i.eps' % (sp,scaler,n_neuron),format='eps')
plt.show(block=False)
# %%
a=(y_test_df[sp] - predict_df[sp]) / y_test_df[sp]
test_data=pd.concat([X_test_df,y_test_df],axis=1)
pred_data=pd.concat([X_test_df,predict_df],axis=1)

test_data.to_hdf('sim_check.H5',key='test')
pred_data.to_hdf('sim_check.H5',key='pred')

# Save model
sess = K.get_session()
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, './exported/my_model')
model.save('FPV_ANN_tabulated_%s.H5' % scaler)

# write the OpenFOAM ANNProperties file
writeANNProperties(in_scaler,out_scaler,scaler)

# Convert the model to
#run -i k2tf.py --input_model='FPV_ANN_tabulated_Standard.H5' --output_model='exported/FPV_ANN_tabulated_Standard.pb'
