import numpy as np
from sklearn import preprocessing

import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt
#%matplotlib inline

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import r2_score

'''
Environment to work with different ANN architectures and CoolProp 
'''

class ANN_realgas_toolbox(object):
    def __init__(self):
        self.history = None
        self.predictions = None
        self.T_test = None
        self.rho_TP_train = None
        self.rhoTP_scaler = None
        self.T_scaler = None
        self.T_P_train = None
        self.rho_vec= None
        self.p_vec = None
        self.model = None
        self.predictions = None
        self.callbacks_list = None
        self.fluid= None
        self.P_max = None
        self.P_min = None
        self.nP = None
        self.T_max = None
        self.T_min = None
        self.nT = None
        self.T_vec = None
        self.p_c = None
        self.T_P_test = None
        self.rho_predict = None
        self.rho_test = None
        self.test_points = None

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


    def genTrainData(self, nT=100, T_min=100, T_max=160, nP=100, P_min=100, P_max=2, fluid='nitrogen'):
        ''' 
        Generates the test data from coolprop. 
        input: nT, T_min, T_max, np, P_min, P_max and the fluid
        '''
        ######################
        self.fluid= fluid
        self.P_max = P_max
        self.P_min = P_min
        self.nP = nP
        self.T_max = T_max
        self.T_min = T_min
        self.nT = nT

        print('Generate data ...')
        # n_train = 20000

        n_train = nT * nP

        # get critical pressure
        self.p_c = CP.PropsSI(fluid, 'pcrit')

        self.p_vec = np.linspace(1, 3, nP) * self.p_c
        self.T_vec = np.linspace(T_min, T_max, nT)
        self.rho_vec = np.asarray([CP.PropsSI('D', 'T', x, 'P', 1.1 * self.p_c, fluid) for x in self.T_vec])

        # prepare input
        # rho = f(T, P)
        # 1. uniform random
        self.T_P_train = np.random.rand(n_train, 2)
        # 2. family curves
        # T_P_train = np.random.rand(n_train, 1)
        # tmp = np.ones((nT, nP))* np.linspace(0, 1, nP)
        # T_P_train = np.append(T_P_train, tmp.reshape(-1, 1), axis=1)

        self.rho_TP_train = np.asarray(
            [self.rho_TP_gen(x, fluid) for x in (self.T_P_train * [(self.T_max - self.T_min), (self.P_max - self.P_min) * self.p_c] + [self.T_min, self.p_c])])

        # normalize train data
        self.rhoTP_scaler = preprocessing.MinMaxScaler()
        self.rho_TP_train = self.rhoTP_scaler.fit_transform(self.rho_TP_train.reshape(-1, 1))

        # normalize test data
        self.T_scaler = preprocessing.MinMaxScaler()
        self.T_test = self.T_scaler.fit_transform(self.T_vec.reshape(-1, 1))

    ######################################
    # different ANN model types
    def setResnet(self, indim=2, n_neurons=200, loss='mse',optimizer='adam', batch_norm=False):
        '''default settings: resnet'''
        ######################
        print('set up Resnet ANN')
        self.model = None

        # This returns a tensor
        inputs = Input(shape=(indim,))

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(n_neurons, activation='relu')(inputs)
        # less then 2 res_block, there will be variance
        x = self.res_block(x, n_neurons, stage=1, block='a', bn=batch_norm)
        x = self.res_block(x, n_neurons, stage=1, block='b', bn=batch_norm)
        # last outout layer with linear activation function
        self.predictions = Dense(1, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=self.predictions)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # checkpoint (save the best model based validate loss)
        try:
            filepath = "./weights.best.hdf5"
        except:
            print('check your fielpath')

        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=10)
        self.callbacks_list = [checkpoint]


    def setSequential(self, indim=2, hiddenLayer=3,n_neurons=200, loss='mse', optimizer='adam', batch_norm=False):
        ######################
        print('set up Sequential (MLP) ANN')
        self.model = None

        self.model = Sequential()
        self.model.add(Dense(n_neurons, input_dim= indim, activation='relu'))
        # create the hidden layers
        for l in range(hiddenLayer):
            self.model.add(Dense(n_neurons, init='uniform', activation='relu'))
        self.model.add(Dense(units=1, activation='linear'))

        # model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # checkpoint (save the best model based validate loss)
        try:
            filepath = "./weights.best.hdf5"
        except:
            print('check your fielpath')

        checkpoint = ModelCheckpoint(filepath,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min',
                                     period=10)
        self.callbacks_list = [checkpoint]


    def fitModel(self, batch_size=1024, epochs=400, vsplit=0.1):
        # fit the model
        self.history = self.model.fit(
            # T_train, rho_train,
            self.T_P_train, self.rho_TP_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=vsplit,
            verbose=2,
            callbacks=self.callbacks_list,
            shuffle=True)


    def predict(self):
        # loads the best weights
        self.model.load_weights("./weights.best.hdf5")

        self.rho_predict = []
        self.rho_test = []
        try:
            for x_p in self.test_points:
                self.T_P_test = np.append(self.T_test, np.ones((len(self.T_test), 1)) * (x_p - self.P_min) / (self.P_max - self.P_min), axis=1)
                predict = self.model.predict(self.T_P_test)
                rho_ref = np.asarray([CP.PropsSI('D', 'T', x_T, 'P', x_p * self.p_c, self.fluid) for x_T in self.T_vec])
                rho_ref = self.rhoTP_scaler.transform(rho_ref.reshape(-1, 1))

                self.rho_predict.append(predict)
                self.rho_test.append(rho_ref)
        except:
            print('No test points defined!')

        plt.show(block=False)


    def plotLoss(self):
        #######################
        # plot loss
        #######################
        fig = plt.figure(1)
        plt.semilogy(self.history.history['loss'])
        #if vsplit:
        plt.semilogy(self.history.history['val_loss'])
        plt.title('mse')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show(block=False)


    def plotPredict(self):
        #######################
        # 1.Plot actual vs prediction for training set
        #######################
        fig = plt.figure(2)
        for prdt, ref in zip(self.rho_predict, self.rho_test):
            plt.plot(self.T_scaler.inverse_transform(self.T_test), self.rhoTP_scaler.inverse_transform(prdt), 'b-')
            plt.plot(self.T_scaler.inverse_transform(self.T_test), self.rhoTP_scaler.inverse_transform(ref), 'r:')
        plt.legend(['predict', 'CoolProp'], loc='upper right')
        plt.title(self.test_points)
        plt.show(block=False)

    def plotAccuraccy(self):
        #######################
        # 2.L2 accuracy plot
        # Compute R-Square value for training set
        #######################

        a = np.asarray(self.rho_predict).reshape(-1, 1)
        b = np.asarray(self.rho_test).reshape(-1, 1)
        TestR2Value = r2_score(a, b)
        print("Training Set R-Square=", TestR2Value)

        fig = plt.figure(3)
        plt.plot(a, b, 'k^', ms=3, mfc='none')
        plt.title('R2 =' + str(TestR2Value))

        plt.show(block=False)


    def setTestPoints(self,vector=[1.02 ,1.05, 1.1,1.2,1.5], random= False, points = 5):

        if random:
            self.test_points = np.random.random(points)

        else:
            self.test_points = vector





