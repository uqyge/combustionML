import numpy as np
from sklearn import preprocessing

import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras import layers
from keras.callbacks import ModelCheckpoint


def res_block(input_tensor, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Dense(100, name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)

    x = Dense(100, name=conv_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


######################
print('Generate data ...')
# vector length
n_train = 20000

nP = 100

T_min = 100
T_max = 160

fluid = 'nitrogen'

# get critical pressure
p_c = CP.PropsSI(fluid, 'pcrit')

p_vec = np.linspace(1, 3, nP) * p_c
T_vec = np.linspace(T_min, T_max, n_train)
rho_vec = np.asarray([CP.PropsSI('D', 'T', x, 'P', 1.1 * p_c, fluid) for x in T_vec])

T_train = np.random.rand(n_train) * (T_max - T_min) + T_min
rho_train = np.asarray([CP.PropsSI('D', 'T', x, 'P', 1.1 * p_c, fluid) for x in T_train])

# normalize
T_scaler = preprocessing.MinMaxScaler()
T_train = T_scaler.fit_transform(T_train.reshape(-1, 1))
T_test = T_scaler.transform(T_vec.reshape(-1, 1))

rho_scaler = preprocessing.MinMaxScaler()
rho_train = rho_scaler.fit_transform(rho_train.reshape(-1, 1))
rho_test = rho_scaler.transform(rho_vec.reshape(-1, 1))
#rho_norm = rho_vec


######################
print('set up ANN')
# ANN parameters
dim = 1
batch_size = 512
epochs = 1000
vsplit = 0.01

# model = Sequential()
# model.add(Dense(100, activation='relu', input_dim=1, init='uniform'))
# model.add(Dense(100, activation='relu', init='uniform'))
# model.add(Dense(100, activation='relu', init='uniform'))
# model.add(Dense(100, activation='relu', init='uniform'))
# model.add(Dense(output_dim=1, activation='linear'))

# This returns a tensor
inputs = Input(shape=(dim,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(100, activation='relu')(inputs)

x = res_block(x, stage=1, block='a')
x = res_block(x, stage=1, block='b')
x = res_block(x, stage=1, block='c')

predictions = Dense(1, activation='linear')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
filepath = "./tmp/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=10)
callbacks_list = [checkpoint]

# fit the model
history = model.fit(T_train, rho_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=vsplit,
                    verbose=2,
                    callbacks=callbacks_list,
                    shuffle=True)


######################
print('model predict')
model.load_weights("./tmp/weights.best.hdf5")
predict = model.predict(T_test)


######################
print('post processing')
# 1.Plot actual vs prediction for training set
plt.plot(T_test, predict)
plt.plot(T_test, rho_test)
plt.show()

# 2.Plot actual vs prediction for training set
fig = plt.figure()
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 3.Plot actual vs prediction for training set
fig = plt.figure()
plt.plot(predict, rho_test, 'ro')
# Compute R-Square value for training set
from sklearn.metrics import r2_score

TestR2Value = r2_score(predict, rho_test)
print("Training Set R-Square=", TestR2Value)
