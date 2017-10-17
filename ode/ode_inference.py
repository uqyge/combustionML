# load and use weights from a checkpoint
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

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

# load weights
model.load_weights("./tmp/weights.best.hdf5")

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])


print("Created model and loaded weights from file")


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

surface_predict = model.predict(x_test).reshape(ny, nx)

# predicted surface
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


# error surface
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