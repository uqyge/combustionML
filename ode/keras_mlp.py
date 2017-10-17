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
from keras.callbacks import ModelCheckpoint


def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))


nx = 30
ny = 30

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)

x_test = np.zeros((ny*nx, 2))
surface = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        x_test[i*nx+j] = [x, y]
        surface[i][j] = analytic_solution([x, y])

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
print('generate sample data from analytic solution')
x_bc_l = np.asarray([[0, y] for y in y_space])
x_bc_r = np.asarray([[1, y] for y in y_space])
x_bc_b = np.asarray([[x, 0] for x in x_space])
x_bc_t = np.asarray([[x, 1] for x in x_space])

n_train = 512 * 2

x_train = np.random.rand(n_train, 2)
x_train = np.concatenate((x_train, x_bc_l, x_bc_r, x_bc_b, x_bc_t))

y_train = [analytic_solution(x) for x in x_train]
y_train = np.reshape(np.asarray(y_train), (-1, 1))


print('Building model...')
batch_size = 256 * 2
epochs = 3000
vsplit = 0.01


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

# compile model
from keras import optimizers
adam = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.99)

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
    verbose=1,
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


# test
surface_predict = model.predict(x_test).reshape(ny, nx)

### predicted surface
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

### error surface
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf_error = ax.plot_surface(X, Y, abs(surface_predict - surface), rstride=1, cstride=1, cmap=cm.viridis,
                             linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 0.1)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.colorbar(surf_error)
