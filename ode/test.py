'''
keras mlp regression
'''
from __future__ import print_function

import numpy as np
nx = 10
ny = 10

x_space = np.linspace(0, 1, nx)
y_space = np.linspace(0, 1, ny)


def analytic_solution(x):
    return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
           np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))




x_input = np.zeros((2, nx*ny))
y_anal = np.zeros((nx*ny,))

surface = np.zeros((ny, nx))
for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])
        print(i * len(x_space) + j)
        x_input[:,i * len(x_space) + j] = [x, y]
        y_anal[i*len(x_space) + j] = analytic_solution([x, y])

x_input = x_input.transpose()
y_anal = y_anal.transpose()
print('generate data from analytic solution')
