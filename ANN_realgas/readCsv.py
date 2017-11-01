import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
from sklearn import preprocessing

from keras.models import Model
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint

import cntk


def res_block(input_tensor, n_neuron, stage, block, bn=False):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Dense(n_neuron, name=conv_name_base + '2a')(input_tensor)
    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Dense(n_neuron, name=conv_name_base + '2b')(x)
    if bn:
        x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


df = pd.DataFrame()
path = 'data'
for fn in os.listdir('./data'):
    # if os.path.isfile(fn):
    print(fn)
    tmp = pd.read_csv(os.path.join(path, fn))
    tmp['p'] = int(fn.replace('bar.csv', ''))
    df = df.append(tmp)

p_scaler = preprocessing.MinMaxScaler()
p=p_scaler.fit_transform(df['p'].values.reshape(-1, 1))
he_scaler = preprocessing.MinMaxScaler()
he=he_scaler.fit_transform(df['he'].values.reshape(-1, 1))

rho_scaler = preprocessing.MinMaxScaler()
rho=rho_scaler.fit_transform(df['rho'].values.reshape(-1, 1))
T_scaler = preprocessing.MinMaxScaler()
T=T_scaler.fit_transform(df['T'].values.reshape(-1, 1))
mu_scaler = preprocessing.MinMaxScaler()
mu=mu_scaler.fit_transform(df['thermo:mu'].values.reshape(-1, 1))
cp_scaler = preprocessing.MinMaxScaler()
cp=cp_scaler.fit_transform(df['Cp'].values.reshape(-1, 1))

x_train = np.append(
    p,
    he,
    axis=1)
y_train = np.concatenate(
    (rho,T,mu,cp),
    axis=1
)

######################
print('set up ANN')
# ANN parameters
dim_input = 2
dim_label = y_train.shape[1]
n_neuron = 100
batch_size = 1024
epochs = 4000
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


#########################################
model.load_weights("./tmp/weights.best.cntk.hdf5")

cntk.combine(model.outputs).save('mayerTest.dnn')


ref=df.loc[df['p']==34]
x_test = np.append(
    p_scaler.transform(ref['p'].values.reshape(-1, 1)),
    he_scaler.transform(ref['he'].values.reshape(-1, 1)),
    axis=1)

predict = model.predict(x_test.astype(float))
plt.figure()
plt.plot(ref['T'],ref['rho'],'r:')
plt.plot(ref['T'],rho_scaler.inverse_transform(predict[:,0].reshape(-1,1)),'b-')
plt.figure()
plt.plot(ref['T'],ref['thermo:mu'],'r:')
plt.plot(ref['T'],mu_scaler.inverse_transform(predict[:,2].reshape(-1,1)),'b-')
plt.figure()
plt.plot(ref['T'],ref['Cp'],'r:')
plt.plot(ref['T'],cp_scaler.inverse_transform(predict[:,3].reshape(-1,1)),'b-')

print(
    p_scaler.data_min_,
    p_scaler.data_max_,
    p_scaler.data_range_,
    he_scaler.data_min_,
    he_scaler.data_max_,
    he_scaler.data_range_
)