import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, metrics

import matplotlib.pyplot as plt
#%matplotlib inline

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping


'''
Environment to work with different ANN architectures and CoolProp 
'''

class ANN_combustion_Toolbox(object):
    def __init__(self,Train_inp ,Train_out ,Test_inp ,Test_out ):
        self.history = None
        self.predictions = None
        self.model = None
        self.callbacks_list = None
        self.Train_inp = Train_inp
        self.Train_out = Train_out
        self.Test_inp = Test_inp
        self.Test_out = Test_out
        self.columns = list(Test_inp.columns)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predict = None
        self.targets = None
        self.features = None

    def dropTargetColumns(self, cols = ['nut', 'Trace_gradU', 'mag_gradU']):
        try:
            self.Train_out = self.Train_out.drop(cols,axis=1)
            self.Test_out = self.Test_out.drop(cols,axis=1)
        except:
            print('Check data type: should be pandas data frame')


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


    def scale_data(self, features = ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO',
                                           'CO2', 'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T',
                                           'nut', 'Trace_gradU', 'mag_gradU'],
                         targets = ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO','CO2',
                                    'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T']):
        self.targets = targets
        self.features = features

        self.MinMax_X = preprocessing.MinMaxScaler()
        self.MinMax_y = preprocessing.MinMaxScaler()

        self.Train_inp = self.Train_inp[self.features]
        self.Test_inp = self.Test_inp[self.features]
        self.Train_out = self.Train_out[self.targets]
        self.Test_out = self.Test_out[self.targets]

        self.X_train = self.MinMax_X.fit_transform(self.Train_inp)
        self.X_test = self.MinMax_X.fit_transform(self.Test_inp)
        self.y_train = self.MinMax_y.fit_transform(self.Train_out)
        self.y_test = self.MinMax_y.fit_transform(self.Test_out)

        #self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    ######################################
    # different ANN model types
    def setResnet(self, n_neurons=200, blocks=2, loss='mse', optimizer='adam', batch_norm=False):
        '''default settings: resnet'''
        ######################
        outdim = len(self.targets)
        print('set up Resnet ANN')
        self.model = None
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

        # This returns a tensor
        inputs = Input(shape=(len(self.features),))

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(n_neurons, activation='relu')(inputs)
        # less then 2 res_block, there will be variance
        for b in range(blocks):
            x = self.res_block(x, n_neurons, stage=1, block=alphabet[b], bn=batch_norm)
            # x = self.res_block(x, n_neurons, stage=1, block='b', bn=batch_norm)
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
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
        self.callbacks_list = [checkpoint, early_stop]


    def setSequential(self,  hiddenLayer=5,n_neurons=200, loss='mse', optimizer='adam', batch_norm=False):
        ######################
        print('set up Sequential (MLP) ANN')
        self.model = None

        self.model = Sequential()
        self.model.add(Dense(n_neurons, input_dim= len(self.features), activation='relu'))
        # create the hidden layers
        for l in range(hiddenLayer):
            self.model.Dense(n_neurons, activation="relu", kernel_initializer="uniform")
        self.model.add(Dense(units=len(self.targets), activation='linear'))

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
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=vsplit,
                verbose=2,
                callbacks=self.callbacks_list,
                shuffle=True)
        except KeyboardInterrupt:
            print('Training is canceled.')



    def prediction(self):
        # loads the best weights
        self.model.load_weights("./weights.best.hdf5")

        self.predict_y = self.model.predict(self.X_test)


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


    def plotAccuracy(self, target = 'T', all = False):
        #######################
        # 2.L2 accuracy plot
        # Compute R-Square value for test set
        #######################
        column_list = list(self.targets)

        if all:
            for i, tar in enumerate(column_list):
                TestR2Value = metrics.r2_score(self.predict_y[:, i], self.y_test[:, i])
                print("Training Set R-Square=", TestR2Value)

                fig = plt.figure(100+i)
                plt.plot(self.predict_y[:, i], self.y_test[:, i], 'k^', ms=3, mfc='none')
                plt.title('R2 =' + str(TestR2Value) + 'for: '+tar)

        else:

            target_id = [ind for ind, x in enumerate(column_list) if x == target]
            target_id = target_id[0]

            TestR2Value = metrics.r2_score(self.predict_y[:, target_id], self.y_test[:, target_id])
            print("Training Set R-Square=", TestR2Value)

            fig = plt.figure(3)
            plt.plot(self.predict_y[:, target_id], self.y_test[:, target_id], 'k^', ms=3, mfc='none')
            plt.title('R2 =' + str(TestR2Value))

        plt.show(block=False)



    def plotPredict(self, pressure = 20., target = 'T'):
        # resacle your data
        y_test_rescaled = self.MinMax_y.inverse_transform(self.y_test)
        X_test_rescaled = self.MinMax_X.inverse_transform(self.X_test)
        predict_rescaled = self.MinMax_y.inverse_transform(self.predict_y)

        # sort your X data according to ascending pressure
        sort_id = np.argsort(X_test_rescaled[:,0])
        y_sorted = y_test_rescaled[sort_id[::]]
        X_sorted = X_test_rescaled[sort_id[::]]
        predict_sorted = predict_rescaled[sort_id[::]]
        print(predict_sorted)

        vals = pressure
        ix = np.isin(X_sorted[:,0], vals)
        index = np.where(ix)
        index=index[:][0]
        print(index)

        column_list = list(self.targets)

        # find the index of the target label
        target_id=[ind for ind, x in enumerate(column_list) if x==target]
        target_id = target_id[0]
        print(target_id)

        y_for_plot = y_sorted[index,target_id]
        predict_for_plot = predict_sorted[index, target_id]
        print(predict_for_plot)

        y_for_plot.sort()
        predict_for_plot.sort()
        plt.figure(10)
        plt.title('Compare prediction and y_test for field: '+target)
        plt.plot(y_for_plot, '*')
        plt.plot(predict_for_plot, '*')
        plt.legend(['y_test','predict'])
        plt.show(block=False)


    # performs parametric search for the Sequential model
    def gridSearchSequential(self, neurons = [50,100,200,400], layers = [2,3,4,5,6], epochs = [100,200,500], batch= [100,200,500,1000],loss_func = ['mse']):

        best_score = 999999
        for n in neurons:
            for l in layers:
                for lossf in loss_func:
                    self.setSequential(indim=len(self.features), hiddenLayer=l, loss=lossf)
                    for e in epochs:
                        for b in batch:
                            try:
                                self.model.load_weights("./weights.best.hdf5")
                            except:
                                print('No weights yet')
                            # fit the model and apply it to the test set
                            self.fitModel(batch_size=b, epochs=e)
                            self.prediction()
                            R2 = abs(metrics.r2_score(self.predict_y,self.y_test))
                            mse = abs(metrics.mean_squared_error(self.predict_y,self.y_test))
                            print('')
                            print('R2 is: ', R2)
                            print('MSE is: ', mse)
                            #save the best set according to R2 value
                            if mse < best_score:
                                # store the best combination of parameters
                                self.best_set = [n, l, e,b,lossf]
                                best_score= mse#R2
                                self.best_model = self.model
                                # store the best weights separately
                                copyfile("./weights.best.hdf5","./weights.best_R2.hdf5")
        print(' ')
        print('best metrics score: ', best_score)

    # performs parametric search for the Sequential model
    def gridSearchResNet(self, neurons = [400,600,800,1000], layers = [4,5,6,10], epochs = [500,600,800], batch= [1000,2000,5000],loss_func = ['mse']):

        best_score = 999999
        for n in neurons:
            for l in layers:
                for lossf in loss_func:
                    self.setResnet( n_neurons=n, blocks=l, batch_norm=True)
                    for e in epochs:
                        for b in batch:
                            try:
                                self.model.load_weights("./weights.best.hdf5")
                            except:
                                print('No weights yet')
                            # fit the model and apply it to the test set
                            self.fitModel(batch_size=b, epochs=e)
                            self.prediction()
                            R2 = abs(metrics.r2_score(self.predict_y,self.y_test))
                            mse = abs(metrics.mean_squared_error(self.predict_y,self.y_test))
                            print('')
                            print('R2 is: ', R2)
                            print('MSE is: ', mse)
                            #save the best set according to R2 value
                            if mse < best_score:
                                # store the best combination of parameters
                                self.best_set = [n, l, e,b,lossf]
                                best_score= mse#R2
                                self.best_model = self.model
                                # store the best weights separately
                                copyfile("./weights.best.hdf5","./weights.best_R2.hdf5")
        print(' ')
        print('best metrics score: ', best_score)





