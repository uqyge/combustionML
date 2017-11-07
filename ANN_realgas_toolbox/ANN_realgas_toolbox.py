import numpy as np
from sklearn import preprocessing

import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt
#%matplotlib inline

from keras.models import Model
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint

'''
Environment to work with different ANN architectures and CoolProp 
'''

class ANN_realgas_toolkit(object):
    def __init__(self):
        self.history = []
        self.predictions = []
        self.T_test = []
        self.rho_TP_train = []
        self.rhoTP_scaler = []
        self.T_scaler = []
        self.T_P_train = []


    def rho_TP_gen(self,x, fluid):
        rho = CP.PropsSI('D', 'T', x[0], 'P', x[1], fluid)
        return rho

    def res_block(self,input_tensor, n_neuron, stage, block, bn=False):
        ''' creates a resnet (Deep Residual Learning) '''
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Dense(n_neuron, name=conv_name_base + '2a')(input_tensor)
        if bn:
            x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Dense(n_neuron, name=conv_name_base + '2b')(x)
        if bn:
            x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)

        return x


    def genTestData(self, nT=100, T_min=100, T_max=160, nP=100, P_min=1, P_max=2, fluid='nitrogen'):
        ''' 
        Generates the test data from coolprop. 
        input: nT, T_min, T_max, np, P_min, P_max and the fluid
        '''
        ######################
        print('Generate data ...')
        # n_train = 20000

        n_train = nT * nP

        # get critical pressure
        p_c = CP.PropsSI(fluid, 'pcrit')

        p_vec = np.linspace(1, 3, nP) * p_c
        T_vec = np.linspace(T_min, T_max, nT)
        rho_vec = np.asarray([CP.PropsSI('D', 'T', x, 'P', 1.1 * p_c, fluid) for x in T_vec])

        # prepare input
        # rho = f(T, P)
        # 1. uniform random
        self.T_P_train = np.random.rand(n_train, 2)
        # 2. family curves
        # T_P_train = np.random.rand(n_train, 1)
        # tmp = np.ones((nT, nP))* np.linspace(0, 1, nP)
        # T_P_train = np.append(T_P_train, tmp.reshape(-1, 1), axis=1)

        self.rho_TP_train = np.asarray(
            [self.rho_TP_gen(x, fluid) for x in (T_P_train * [(T_max - T_min), (P_max - P_min) * p_c] + [T_min, p_c])])

        # normalize train data
        self.rhoTP_scaler = preprocessing.MinMaxScaler()
        self.rho_TP_train = self.rhoTP_scaler.fit_transform(self.rho_TP_train.reshape(-1, 1))

        # normalize test data
        self.T_scaler = preprocessing.MinMaxScaler()
        self.T_test = self.T_scaler.fit_transform(T_vec.reshape(-1, 1))




