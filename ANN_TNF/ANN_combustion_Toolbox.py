import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

class ANN_combustion_Toolbox(object):
    def __init__(self,Train_inp = Train_arr_inp,Train_out = Train_arr_out,Test_inp = Test_arr_inp,Test_out = Test_arr_out):
        self.history = None
        self.predictions = None
        self.model = None
        self.callbacks_list = None
        self.Train_inp = Train_inp
        self.Train_out = Train_out
        self.Test_inp = Test_inp
        self.Test_out = Test_out
        self.columns = list(Test_inp.columns)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.predict = None

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


    def applyScaler(self):
        scaler_train = MinMaxScaler()
        scaler_test = MinMaxScaler()

        scaler_train.fit(self.Train_inp)
        scaler_test.fit(self.Test_inp)

        print(scaler_test.data_max_)
        self.x_train = scaler_train.transform(self.Train_inp)
        self.y_train = scaler_train.transform(self.Train_out)
        self.x_test = scaler_test.transform(self.Test_inp)
        self.y_test = scaler_test.transform(self.Test_out)

    ######################################
    # different ANN model types
    def setResnet(self, indim=20, outdim=20, n_neurons=200, loss='mse', optimizer='adam', batch_norm=False):
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
        self.predictions = Dense(outdim, activation='linear')(x)

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


    def setSequential(self, indim=20, outdim=20, hiddenLayer=5,n_neurons=200, loss='mse', optimizer='adam', batch_norm=False):
        ######################
        print('set up Sequential (MLP) ANN')
        self.model = None

        self.model = Sequential()
        self.model.add(Dense(n_neurons, input_dim= indim, activation='relu'))
        # create the hidden layers
        for l in range(hiddenLayer):
            self.model.Dense(n_neurons, activation="relu", kernel_initializer="uniform")
        self.model.add(Dense(units=outdim, activation='linear'))

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
        try:
            self.history = self.model.fit(
                # T_train, rho_train,
                self.x_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=vsplit,
                verbose=2,
                callbacks=self.callbacks_list,
                shuffle=True)
        except KeyboardInterrupt:
            print('Training is canceled.')


    def predictModel(self):
        # loads the best weights
        self.model.load_weights("./weights.best.hdf5")

        self.predict = self.model.predict(self.x_test)

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
        plt.legend(['predict', 'ESF'], loc='upper right')
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





