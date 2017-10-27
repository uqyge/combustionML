'''
keras mlp regression
'''
from __future__ import print_function

import numpy as np
from undecorated import undecorated

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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


def bc_dec(func):
    def function_wrapper(x):
        if (x[0] * (x[0] - 1) * x[1] * (x[1] - 1)) != 0:
            bc_trans = (func(x) - x[1] * np.sin(np.pi * x[0])) / (x[0] * (x[0] - 1) * x[1] * (x[1] - 1))
        else:
            bc_trans = x[1] * np.sin(np.pi * x[0]) + (x[0] * (x[0] - 1) * x[1] * (x[1] - 1))
        return bc_trans

    return function_wrapper


# @bc_dec
def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))


nx = 30
ny = 30

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)

print('generate sample data from analytic solution')
x_bc_l = np.asarray([[0, y] for y in y_space])
x_bc_r = np.asarray([[1, y] for y in y_space])
x_bc_b = np.asarray([[x, 0] for x in x_space])
x_bc_t = np.asarray([[x, 1] for x in x_space])

n_train = 1024 * 100

x_train = np.random.rand(n_train, 2)
x_train = np.concatenate((x_train, x_bc_l, x_bc_r, x_bc_b, x_bc_t))

y_train = [analytic_solution(x) for x in x_train]
y_train = np.reshape(np.asarray(y_train), (-1, 1))

print('Building model...')
batch_size = 1024
epochs = 500
vsplit = 0.01


# model = Sequential()
# model.add(Dense(100, input_shape=(2,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.))
#
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.))
#
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.))
#
# model.add(Dense(100))
# model.add(Activation('relu'))
# model.add(Dropout(0.))
#
# model.add(Dense(1))
# #model.add(Activation('relu'))
# model.add(Activation('linear'))


# This returns a tensor
inputs = Input(shape=(2,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(100, activation='relu')(inputs)

x = res_block(x,stage=1,block='a')
x = res_block(x,stage=2,block='b')

predictions = Dense(1, activation='linear')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

# compile model
from keras import optimizers

adam = optimizers.Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
# filepath = "./tmp/weights-improvement-{epoch:02d}-{val_loss:.2e}.hdf5"
filepath = "./tmp/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=10)
callbacks_list = [checkpoint]

# fit the model
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_split=vsplit,
    callbacks=callbacks_list)

# score = model.evaluate(x_test, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
if vsplit:
    # summarize history for loss
    fig = plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('mse')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# visualisation
# 1. analytical solution
x_test = np.zeros((ny * nx, 2))
surface = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        x_test[i * nx + j] = [x, y]
        surface[i][j] = undecorated(analytic_solution)([x, y])
###
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
                       linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(surf)

# 2.test solution
surface_predict = model.predict(x_test).reshape(ny, nx)

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf_pdt = ax.plot_surface(X, Y, surface_predict, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(surf_pdt)

# 3.error surface
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf_error = ax.plot_surface(X, Y, abs(surface_predict - surface), rstride=1, cstride=1, cmap=cm.viridis,
                             linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 0.01)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(surf_error)

# 4.Plot actual vs prediction for training set
fig = plt.figure()
plt.plot(surface.reshape(-1), surface_predict.reshape(-1), 'ro')
# Compute R-Square value for training set
from sklearn.metrics import r2_score

TestR2Value = r2_score(surface.reshape(-1), surface_predict.reshape(-1))
print("Training Set R-Square=", TestR2Value)
