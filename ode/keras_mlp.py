'''
keras mlp regression
'''
from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


nx = 10
ny = 10

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)


def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))


surface = np.zeros((ny, nx))


for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])


# batch_size = 20
# epochs = 5
#
# print('generate data from analytic solution')
# (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words,
#                                                          test_split=0.2)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
#
# print(num_classes, 'classes')
#
# print('Vectorizing sequence data...')
# tokenizer = Tokenizer(num_words=max_words)
# x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
# x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
#
# print('Convert class vector to binary class matrix '
#       '(for use with categorical_crossentropy)')
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
# print('y_train shape:', y_train.shape)
# print('y_test shape:', y_test.shape)
#
# print('Building model...')
# model = Sequential()
# model.add(Dense(10, input_shape=(max_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
#
#
# model.compile(loss='mean_squared_error',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)
# score = model.evaluate(x_test, y_test,
#                        batch_size=batch_size, verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])