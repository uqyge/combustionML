'''
keras mlp regression
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


nx = 10
ny = 10

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)


def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))




x_input = np.zeros((ny,nx,2))
#y_anal = np.zeros((nx*ny,))

surface = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])
        x_input[i][j] = [x, y]
        # print(i * len(x_space) + j)
        # x_input[:,i * len(x_space) + j] = [x, y]
        # y_anal[i*len(x_space) + j] = analytic_solution([x, y])

x_input = x_input.reshape(-1, x_input.shape[-1])
#x_input = x_input.transpose()
y_anal = surface.reshape(-1,1)
print('generate data from analytic solution')

batch_size = 10
epochs = 5000


print('Building model...')
model = Sequential()
model.add(Dense(10, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0))
model.add(Dense(1))


model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_input, y_anal,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
# score = model.evaluate(x_test, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()