import math
import random

import matplotlib.pyplot as plt
import numpy
from matplotlib import pyplot

numpy.random.seed(7)
random.seed = 775
print(type(random.seed))
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

THEANO_FLAGS = ""

'''
Some comments: 
it's important to have two layers
avoid sigmoid in the last layer: activation='linear'

'''


numpy.array
r = Sequential()

def setup_nn():

    r.add(Dense(50, activation='relu', input_dim=1, init='uniform'))
    r.add(Dense(70, activation='relu',init='uniform'))
    r.add(Dense(output_dim=1, activation='linear'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=False)
    r.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])


def target_function(X):
    a = math.sin(X)

    return a*10


def trainRandomX(samplesize):
    X = []
    Y = []
    for j in range(0, samplesize):
        xj = random.random()+random.random()+random.random()+random.random()+random.random() #random.random()*2*math.pi
        X.append(xj)
        Y.append(target_function(xj))
    # X=numpy.array(X)
    # Y=numpy.array(Y)
    r.fit(X, Y, batch_size=10, nb_epoch=20)

    return


def testRandomX():
    X = [random.random()+random.random()+random.random()+random.random()+random.random()]
    Y = target_function(X[0])
    X = numpy.array(X)
    Ypred = r.predict(X, batch_size=10)
    error = Ypred[0][0] - Y
    print("error: ", error)
    # print(Ypred)
    return [X, Ypred[0][0]]


setup_nn()
plt.interactive(False)

# for i in range(0, 1):
trainRandomX(10000)

error = 0
X = []
Y = []



def plotfunction():
    X = []
    Y = []
    for i in range(0, 500):
        x = i / 100
        X.append(x)
        Y.append(target_function(x))
    pyplot.plot(X, Y, '.r')


plotfunction()
for i in range(0, 100):
    # error += abs(testRandomX())
    XY = testRandomX()
    X.append(XY[0][0])
    Y.append(XY[1])
pyplot.plot(X, Y, 'ob')

print("average error: ", error / 20)
plt.show()