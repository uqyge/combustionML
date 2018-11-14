#/usr/bin/python3

import numpy as np
import datetime
from sklearn import preprocessing, metrics, model_selection
import xgboost as xgb
import os
from shutil import copyfile
from sklearn.externals import joblib


try:
    import matplotlib.pyplot as plt
except:
    print('plotting is not possible on the cluster')
#%matplotlib inline

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.multioutput import MultiOutputRegressor

try:
    import cntk
except:
    print('cntk toolkit is not available!')

from abc import ABC, abstractmethod

'''
This is an adaptive machine learning toolbox to be used for data analysis in combustion and real-gas thermodynamics 
at LRT10/UniBW

For the moment it supports:
- MLP
- ResNet
- xgboost 

@author: mhansinger
'''

class ML_Toolbox(ABC):
    # this is the abstract base class for the machine leraning toolbox
    def __init__(self):
        self.history = None
        self.predictions = None
        self.model = None
        self.predictions = None
        self.callbacks_list = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.targets = None
        self.features = None
        self.best_set = []
        self.best_model = None
        self.predict_for_plot = None
        self.y_for_plot = None
        self.X_test_for_plot = None
        self.MinMax_X = None
        self.MinMax_y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predict = None
        self.targets = None
        self.features = None

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


    ######################################
    # ResNet model
    def setResnet(self, n_neurons=200, blocks = 2, loss='mse', optimizer='adam', batch_norm=False):
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
            #x = self.res_block(x, n_neurons, stage=1, block='b', bn=batch_norm)
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
        self.callbacks_list = [checkpoint,early_stop]


    #####################################
    # sequential model
    def setSequential(self, indim = 2, hiddenLayer = 4,n_neurons = 200, loss = 'mse', optimizer = 'adam', batch_norm=False):
        ######################
        outdim = len(self.targets)
        print('set up Sequential (MLP) ANN')
        self.model = None

        self.model = Sequential()
        self.model.add(Dense(n_neurons, input_dim= indim, activation='relu'))

        # create the hidden layers
        for l in range(hiddenLayer):
            self.model.add(Dense(n_neurons, init='uniform', activation='relu'))
        self.model.add(Dropout(0.2))
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
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
        self.callbacks_list = [checkpoint, early_stop]

    # fit the model!
    def fitModel(self, batch_size=1024, epochs=400, vsplit=0.1,  shuffle=True):
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
            print('\nYou canceled the training.')

    def prediction(self):
        # loads the best weights
        self.model.load_weights("./weights.best.hdf5")
        self.predict_y = self.model.predict(self.X_test)


    def plotLoss(self):
        #plots the loss
        fig = plt.figure(1)
        plt.semilogy(self.history.history['loss'])
        # if vsplit:
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


    # performs parametric search for the Sequential model
    def gridSearchSequential(self, neurons = [200,400,500], layers = [6,10,20], epochs = [100,200,500], batch= [500,1000],loss_func = ['mse']):

        best_score = 999999
        for n in neurons:
            for l in layers:
                for lossf in loss_func:
                    self.setSequential(indim=2, n_neurons=n, hiddenLayer=l, loss=lossf)
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
    def gridSearchResNet(self, neurons = [200,400,600], layers = [3,4,5], epochs = [500,600,800], batch= [1000,2000], loss_func = ['mse']):

        best_score = 999999
        for n in neurons:
            for l in layers:
                for lossf in loss_func:
                    self.setResnet(indim=2, n_neurons=n, blocks=l, batch_norm=True)
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


    # def save_model(self,name='best_model.pkl'):
    #     #stores the model!
    #     joblib.dump(self.model, name)
    #
    # def load_model(self,name):
    #     # loads a previous model
    #     self.model= joblib.load(name)

    @abstractmethod
    def scale_data(self):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def plotPredict(self):
        raise NotImplementedError("Please Implement this method")



###############################################################

class ANN_realgas_toolbox(ML_Toolbox):
    def __init__(self):
        super().__init__()

        self.MinMax_X = []
        self.predict_y = None
        self.test_points = None
        self.data_dict = None
        self.all_data = None
        self.y = None
        self.X = None
        self.multiReg = None
        self.predict_for_plot = None
        self.y_for_plot = None
        self.X_test_for_plot = None
        self.T_for_X = None


    def import_data(self):
        path = input('Where is your data? ')
        print('Importing the data...')
        # data files
        files = os.listdir(path)
        # check if csv! and sort
        files = [f for f in files if f[-3:]=='csv']
        files.sort()

        self.data_dict = {}
        for file in files:
            pressure = file[0:2]
            df = pd.read_csv(path+file)
            df['p'] = float(pressure)
            self.data_dict['data_p' + pressure] = df

        del df

        print('Your data dictionary has the following keys: ', self.data_dict.keys())
        print('Now reshaping to one data frame')
        keys = list(self.data_dict.keys())
        keys.sort()
        self.all_data = self.data_dict[keys[0]]
        for num in range(1,len(keys)):
            print(keys[num])
            df = self.data_dict[keys[num]]
            #print(df)
            self.all_data=self.all_data.append(df, ignore_index = True)
            del df

        print('You have ' + str(len(self.all_data)) + ' data points')
        use_cols = list(self.all_data.columns)
        self.all_data = self.all_data[use_cols[1:]]


    def scale_data(self, features = ['p','he'], targets = ['rho','T','thermo:psi','thermo:mu','thermo:alpha','thermo:as','thermo:Z','Cp'], test_size = 0.3, random=42):
        '''
        
        :param features: choose your features
        :param targets:  choose your targests
        :return:         scaled and splitted train and test data sets: X_train, X_test, y_train, y_test
        '''
        self.import_data()

        self.targets = targets
        self.features = features
        self.X = self.all_data[features]
        #self.X = np.array(X)
        self.y = self.all_data[targets]
        #self.y = np.array(y)

        self.MinMax_X = preprocessing.MinMaxScaler()
        self.MinMax_y = preprocessing.MinMaxScaler()

        self.X = self.MinMax_X.fit_transform(self.X)
        self.y = self.MinMax_y.fit_transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            model_selection.train_test_split(self.X, self.y, test_size=test_size, random_state=random)


    def plotPredict(self, pressure = 40., target = 'T'):
        # resacle your data
        y_test_rescaled = self.MinMax_y.inverse_transform(self.y_test)
        X_test_rescaled = self.MinMax_X.inverse_transform(self.X_test)
        predict_rescaled = self.MinMax_y.inverse_transform(self.predict_y)

        # sort your X data according to ascending pressure
        sort_id = np.argsort(X_test_rescaled[:,0])
        y_sorted = y_test_rescaled[sort_id[::]]
        X_sorted = X_test_rescaled[sort_id[::]]
        predict_sorted = predict_rescaled[sort_id[::]]
        #print(predict_sorted)

        vals = pressure
        ix = np.isin(X_sorted[:,0], vals)
        index = np.where(ix)
        index=index[:][0]
        #print(index)

        column_list = list(self.targets)

        # find the index of the target label
        target_id=[ind for ind, x in enumerate(column_list) if x==target]
        target_id = target_id[0]
        #print(target_id)

        self.y_for_plot = y_sorted[index,target_id]
        self.predict_for_plot = predict_sorted[index, target_id]
        self.X_test_for_plot = X_sorted[index, 1]   # needs to be the second column for enthalpy
        self.T_for_X = y_sorted[index,1]

        plt.figure(10)
        plt.title('Compare prediction and y_test for field: '+target)
        plt.plot(self.T_for_X,self.y_for_plot, 'ok')
        plt.plot(self.T_for_X,self.predict_for_plot, '^r')
        plt.legend(['y_test','y_predict'])
        plt.xlabel('T [K]')
        plt.ylabel(target)
        plt.show(block=False)


    def writDNN(self,name=''):
        '''Write a .dnn file to use it with OpenFoam'''
        if name == '':
            print('Give it a name!')
        else:
            cntk.combine(self.model.outputs).save(name)


    # XGBoost classifier!
    def xgboost(self,depth = 5):
        tuned_params = {"objective":"reg:linear",'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': depth}
        self.multiReg = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear', params = tuned_params))
        self.multiReg.fit(self.X_train,self.y_train)
        self.predict_y = self.multiReg.predict(self.X_test)
        score = metrics.r2_score(self.predict_y, self.y_test)
        print('The R2 score from XGBoost is: ',score)


##################################################

class ANN_combustion_Toolbox(ML_Toolbox):
    def __init__(self,Train_inp ,Train_out ,Test_inp ,Test_out ):
        super().__init__()

        self.Train_inp = Train_inp
        self.Train_out = Train_out
        self.Test_inp = Test_inp
        self.Test_out = Test_out
        self.columns = list(Test_inp.columns)


    def dropTargetColumns(self, cols = ['nut', 'Trace_gradU', 'mag_gradU']):
        try:
            self.Train_out = self.Train_out.drop(cols,axis=1)
            self.Test_out = self.Test_out.drop(cols,axis=1)
        except:
            print('Check data type: should be pandas data frame')


    def scale_data(self, features = ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO',
                                           'CO2', 'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T',
                                           'nut', 'Trace_gradU', 'mag_gradU'],
                         targets = ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO','CO2',
                                    'H', 'H2', 'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T']):
        self.targets = targets
        self.features = features

        self.MinMax_X = preprocessing.MinMaxScaler()
        self.MinMax_y = preprocessing.MinMaxScaler()

        #self.MinMax_X = preprocessing.StandardScaler()
        #self.MinMax_y = preprocessing.StandardScaler()

        self.Train_inp = self.Train_inp[self.features]
        self.Test_inp = self.Test_inp[self.features]
        self.Train_out = self.Train_out[self.targets]
        self.Test_out = self.Test_out[self.targets]

        # fit the scalers to the whole data
        X_data = self.Train_inp.copy()
        X_data = X_data.append(self.Test_inp)

        y_data = self.Train_out.copy()
        y_data = y_data.append(self.Test_out)

        self.MinMax_X.fit(X_data)
        self.MinMax_y.fit(y_data)

        self.X_train = self.MinMax_X.transform(self.Train_inp)
        self.X_test = self.MinMax_X.transform(self.Test_inp)
        self.y_train = self.MinMax_y.transform(self.Train_out)
        self.y_test = self.MinMax_y.transform(self.Test_out)



    def plotPredict(self, target = 'T'):
        # resacle your data
        y_test_rescaled = self.MinMax_y.inverse_transform(self.y_test)
        X_test_rescaled = self.MinMax_X.inverse_transform(self.X_test)
        predict_rescaled = self.MinMax_y.inverse_transform(self.predict_y)

        column_list = list(self.targets)

        # find the index of the target label
        target_id=[ind for ind, x in enumerate(column_list) if x==target]
        target_id = target_id[0]

        self.y_for_plot = y_test_rescaled[:,target_id]
        self.predict_for_plot = predict_rescaled[:, target_id]
        self.X_test_for_plot = X_test_rescaled[:, 1]   # needs to be the second column for enthalpy

        plt.figure(10)
        plt.title('Compare prediction and y_test for field: '+target)
        plt.plot(self.y_for_plot, 'ok')
        plt.plot(self.predict_for_plot, '^r')
        plt.legend(['y_test','y_predict'])
        #plt.xlabel('T [K]')
        plt.ylabel(target)
        plt.show(block=False)




