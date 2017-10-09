'''
keras mlp regression
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))


ntrain = 512 * 2
nx = 30
ny = 30

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)

x_input = np.zeros((ny, nx, 2))

surface = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])
        x_input[i][j] = [x, y]

x_input = x_input.reshape(-1, x_input.shape[-1])
y_anal = surface.reshape(-1, 1)

print('generate data from analytic solution')
x_bc_l = np.asarray([[0, y] for y in y_space])
x_bc_r = np.asarray([[1, y] for y in y_space])
x_bc_b = np.asarray([[x, 0] for x in x_space])
x_bc_t = np.asarray([[x, 1] for x in x_space])

x_train = np.random.rand(int(ntrain / 2), 2)
# x_train = 1 - x_train
x_train = np.concatenate((x_train, 1 - x_train))
x_train = np.concatenate((x_train, x_bc_l, x_bc_r, x_bc_b, x_bc_t))

y_train = [analytic_solution(x) for x in x_train]
# y_train = np.reshape(np.asarray(y_train), (ntrain+100, -1))
y_train = np.reshape(np.asarray(y_train), (-1, 1))

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

###
batch_size = 256 * 2
epochs = 3000
vsplit = 0.01

print('Building model...')
model = Sequential()
model.add(Dense(100, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.))
# model.add(Dense(200))
# model.add(Activation('relu'))
# model.add(Dropout(0.))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.))
model.add(Dense(1))

from keras import optimizers

adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_split=vsplit)
# score = model.evaluate(x_test, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
if vsplit:
    # # summarize history for accuracy
    # fig = plt.figure()
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

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

x_test = x_input + 0.0
x_test_space = x_test.reshape(ny, nx, 2)[0, :, 1]
y_test_space = x_test.reshape(ny, nx, 2)[:, 0, 0]
surface_predict = model.predict(x_test).reshape(ny, nx)

### predicted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_test_space, y_test_space)
surf_pdt = ax.plot_surface(X, Y, surface_predict, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(surf_pdt)

### error surface
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_test_space, y_test_space)
surf_error = ax.plot_surface(X, Y, abs(surface_predict - surface), rstride=1, cstride=1, cmap=cm.viridis,
                             linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 0.1)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(surf_error)
