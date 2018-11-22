import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

from resBlock import res_block
from pre import read_csv_data

# %%
x_train, y_train, df, in_scaler, out_scaler = read_csv_data('data')

######################
print('set up ANN')
# ANN parameters
dim_input = 2
dim_label = y_train.shape[1]
n_neuron = 200
batch_size = 1024
epochs = 40
vsplit = 0.1
batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

# less then 2 res_block, there will be variance
x = res_block(x, n_neuron, stage=1, block='a', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='b', bn=batch_norm)

# x = res_block(x, n_neuron, stage=1, block='c', bn=batch_norm)
# x = res_block(x, n_neuron, stage=1, block='d', bn=batch_norm)

predictions = Dense(dim_label, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

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
    x_train, y_train,
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
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#########################################
model.load_weights("./tmp/weights.best.cntk.hdf5")
# cntk.combine(model.outputs).save('mayerTest.dnn')

# %%
ref = df.loc[df['p'] == 40]
x_test = in_scaler.transform(ref[['p', 'he']])
st = time.time()
predict_val = model.predict(x_test.astype(float), batch_size=1024)
time_gpu_infer = time.time() - st
print('Gpu inference time is ', time_gpu_infer)
# %%
# predict = out_scaler.
sp = 'rho'
predict = pd.DataFrame(out_scaler.inverse_transform(predict_val), columns=['rho', 'T', 'thermo:mu', 'Cp'])
plt.figure()
plt.plot(ref['T'], ref[sp], 'r:')
plt.plot(ref['T'], predict[sp], 'b-')
plt.show()
plt.figure()
plt.plot((ref[sp] - predict[sp]) / ref[sp])
plt.show()

# %%
from keras import backend as K

sess = K.get_session()
saver = tf.train.Saver(tf.global_variables())
saver.save(sess, './exported/my_model')
tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)
np.savetxt('x_test.csv', x_test)
np.savetxt('pred.csv', predict_val)
model.save('mayer.H5')

