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

ntrain = 1000
nx = 20
ny = 20

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)


def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))


x_input = np.zeros((ny,nx,2))

surface = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])
        x_input[i][j] = [x, y]


x_input = x_input.reshape(-1, x_input.shape[-1])
y_anal = surface.reshape(-1,1)

x_train = np.random.rand(ntrain, 2)
x_train = 1 - x_train
y_train = [analytic_solution(x) for x in x_train]
y_train = np.reshape(np.asarray(y_train), (ntrain, -1))
print('generate data from analytic solution')
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

batch_size = 64
epochs = 400
vsplit = 0.1

print('Building model...')
model = Sequential()
model.add(Dense(100, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))




model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
                    x_train, y_train,
    # x_input, y_anal,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=vsplit)
# score = model.evaluate(x_test, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
if vsplit:
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

x_test = x_input+0.0
x_test_space=x_test.reshape(ny,nx,2)[0,:,1]
y_test_space=x_test.reshape(ny,nx,2)[:,0,0]
surface_predict = model.predict(x_test).reshape(ny, nx)

###
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