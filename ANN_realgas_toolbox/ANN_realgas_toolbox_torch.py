import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing, model_selection, metrics


import CoolProp.CoolProp as CP
import os
from shutil import copyfile

import matplotlib.pyplot as plt
#%matplotlib inline

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

# Pytorch
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from tqdm import tqdm

'''
This is going to be the pytorch version of the ANN for real gas  
'''

class ANN_realgas_toolbox_torch(object):
    def __init__(self):
        self.history = None
        self.predictions = None
        self.MinMax_X = []
        self.model = None
        self.predictions = None
        self.callbacks_list = None
        self.predict_y = None
        #self.rho_test = None
        self.test_points = None
        self.data_dict = None
        self.all_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y = None
        self.X = None
        self.multiReg = None
        self.targets = None
        self.features = None
        self.best_set = []
        self.best_model = None
        self.predict_for_plot = None
        self.y_for_plot = None
        self.X_test_for_plot = None
        self.T_for_X = None



    # ## Rewrite that function in pytorch

    # def res_block(self,input_tensor, n_neuron, stage, block, bn=False):
    #     ''' creates a resnet (Deep Residual Learning) '''
    #     conv_name_base = 'res' + str(stage) + block + '_branch'
    #     bn_name_base = 'bn' + str(stage) + block + '_branch'
    #
    #     x = Dense(n_neuron, name=conv_name_base + '2a')(input_tensor)
    #     if bn:
    #         x = BatchNormalization(axis=-1, name=bn_name_base + '2a')(x)
    #     x = Activation('relu')(x)
    #
    #     x = Dense(n_neuron, name=conv_name_base + '2b')(x)
    #     if bn:
    #         x = BatchNormalization(axis=-1, name=bn_name_base + '2b')(x)
    #     x = layers.add([x, input_tensor])
    #     x = Activation('relu')(x)
    #
    #     return x

    def import_data(self,path='data_trax/'):
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


    def scale_split_data(self, features = ['p','he'], targets = ['rho','T','thermo:psi','thermo:mu','thermo:alpha','thermo:as','thermo:Z','Cp']):
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

        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.X, self.y, test_size=0.3, random_state=42)


    # ######################################
    # # different ANN model types
    # def setResnet(self, indim=2, n_neurons=200, blocks = 2, loss='mse', optimizer='adam', batch_norm=False):
    #     '''default settings: resnet'''
    #     ######################
    #     outdim = len(self.targets)
    #     print('set up Resnet ANN')
    #     self.model = None
    #     alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    #
    #     # This returns a tensor
    #     inputs = Input(shape=(indim,))
    #
    #     # a layer instance is callable on a tensor, and returns a tensor
    #     x = Dense(n_neurons, activation='relu')(inputs)
    #     # less then 2 res_block, there will be variance
    #     for b in range(blocks):
    #         x = self.res_block(x, n_neurons, stage=1, block=alphabet[b], bn=batch_norm)
    #         #x = self.res_block(x, n_neurons, stage=1, block='b', bn=batch_norm)
    #     # last outout layer with linear activation function
    #     self.predictions = Dense(outdim, activation='linear')(x)
    #
    #     self.model = Model(inputs=inputs, outputs=self.predictions)
    #     self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    #
    #     # checkpoint (save the best model based validate loss)
    #     try:
    #         filepath = "./weights.best.hdf5"
    #     except:
    #         print('check your fielpath')
    #
    #     checkpoint = ModelCheckpoint(filepath,
    #                                  monitor='val_loss',
    #                                  verbose=1,
    #                                  save_best_only=True,
    #                                  mode='min',
    #                                  period=10)
    #     early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
    #     self.callbacks_list = [checkpoint,early_stop]


    # def setSequential(self, indim=2, hiddenLayer=4,n_neurons=200, loss='mse', optimizer='adam', batch_norm=False):
    #     ######################
    #     outdim = len(self.targets)
    #     print('set up Sequential (MLP) ANN')
    #     self.model = None
    #
    #     self.model = Sequential()
    #     self.model.add(Dense(n_neurons, input_dim= indim, activation='relu'))
    #
    #     # create the hidden layers
    #     for l in range(hiddenLayer):
    #         self.model.add(Dense(n_neurons, init='uniform', activation='relu'))
    #     self.model.add(Dropout(0.2))
    #     self.model.add(Dense(units=outdim, activation='linear'))
    #
    #     # model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    #     self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    #
    #     # checkpoint (save the best model based validate loss)
    #     try:
    #         filepath = "./weights.best.hdf5"
    #     except:
    #         print('check your fielpath')
    #
    #     checkpoint = ModelCheckpoint(filepath,
    #                                  monitor='val_loss',
    #                                  verbose=1,
    #                                  save_best_only=True,
    #                                  mode='min',
    #                                  period=10)
    #     early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='min')
    #     self.callbacks_list = [checkpoint, early_stop]

    def setSequential(self, indim=2, hiddenLayer=4, n_neurons=200, optimizer='adam', device = 'cpu'):
        """
        This is pytorch version
        D_in: input dimension
        H: dimension of hidden layer
        D_out: output dimension
        """
        self.outdim = len(self.targets)

        device = torch.device(device)

        self.model = nn.Sequential(
            nn.Linear(indim, n_neurons),
            nn.Linear(n_neurons,n_neurons),
            nn.Linear(n_neurons, n_neurons),
            nn.Linear(n_neurons, n_neurons),
            nn.Linear(n_neurons, self.outdim)).to(device)

        self.loss_fn = nn.MSELoss(reduction='sum')


    def fitTorch(self,epochs =400,learning_rate = 1e-8):
        print('Starting fitting process')
        self.X_train_tensor = torch.FloatTensor(self.X_train)
        self.y_train_tensor = torch.FloatTensor(self.y_train)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

        # for e in range(epochs):
        #     # Forward pass: compute predicted y by passing x to the model. Module objects
        #     # override the __call__ operator so you can call them like functions. When
        #     # doing so you pass a Tensor of input data to the Module and it produces
        #     # a Tensor of output data.
        #
        #     self.y_pred = self.model(self.X_train_tensor)
        #     loss = self.loss_fn(self.y_pred,self.y_train_tensor)
        #
        #     print(e, loss.item())
        #
        #     # Zero the gradients befor backward prop.
        #     self.model.zero_grad()
        #
        #     # backward pass and update of the
        #     loss.backward()
        #
        #     with torch.no_grad():
        #         for param in self.model.parameters():
        #             param.data -= learning_rate*param.grad

        for epoch in range(epochs):
            epoch += 1
            # inputs = Variable(torch.from_numpy(x_train))
            # labels = Variable(torch.from_numpy(y_correct))

            # clear grads
            optimiser.zero_grad()
            # forward to get predicted values
            outputs = self.model.forward(self.X_train_tensor)
            loss = self.loss_fn(outputs, self.y_train_tensor)
            loss.backward()  # back props
            optimiser.step()  # update the parameters
            print('epoch {}, loss {}'.format(epoch, loss.data[0]))

    # def run(self, epochs, lr, momentum, log_interval=None):
    #     # train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    #     # model = Net()
    #     device = 'cpu'
    #
    #     # # Dataloader implemented
    #     # train_loader = DataLoader(dataset=createDataset(self.X_train,self.y_train), batch_size=10, shuffle=True)
    #     # test_loader = DataLoader(dataset=createDataset(self.X_test, self.y_test),  batch_size=10,
    #     #                           shuffle=False)
    #
    #     train_target = torch.tensor(self.y_train).float()
    #     train = torch.tensor(self.X_train).float()
    #
    #     test_target = torch.tensor(self.y_test).float()
    #     test = torch.tensor(self.X_test).float()
    #
    #     train_tensor = TensorDataset(train, train_target)
    #     train_loader = DataLoader(dataset=train_tensor, batch_size=100, shuffle=True)
    #
    #     test_tensor = TensorDataset(test, test_target)
    #     test_loader = DataLoader(dataset=test_tensor, batch_size=100, shuffle=False)
    #
    #     #if torch.cuda.is_available():
    #     #    device = 'cuda'
    #
    #     optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum)
    #     trainer = create_supervised_trainer(self.model, optimizer, F.nll_loss, device=device)
    #     evaluator = create_supervised_evaluator(self.model,
    #                                             metrics={'accuracy': CategoricalAccuracy(),
    #                                                      'nll': Loss(F.nll_loss)},
    #                                             device=device)
    #
    #     desc = "ITERATION - loss: {:.2f}"
    #     pbar = tqdm(
    #         initial=0, leave=False, total=len(train_loader),
    #         desc=desc.format(0)
    #     )
    #
    #     @trainer.on(Events.ITERATION_COMPLETED)
    #     def log_training_loss(engine):
    #         iter = (engine.state.iteration - 1) % len(train_loader) + 1
    #
    #         if iter % log_interval == 0:
    #             pbar.desc = desc.format(engine.state.output)
    #             pbar.update(log_interval)
    #
    #     @trainer.on(Events.EPOCH_COMPLETED)
    #     def log_training_results(engine):
    #         pbar.refresh()
    #         evaluator.run(test_loader)
    #         metrics = evaluator.state.metrics
    #         avg_accuracy = metrics['accuracy']
    #         avg_nll = metrics['nll']
    #         tqdm.write(
    #             "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #                 .format(engine.state.epoch, avg_accuracy, avg_nll)
    #         )
    #
    #     @trainer.on(Events.EPOCH_COMPLETED)
    #     def log_validation_results(engine):
    #         evaluator.run(test_loader)
    #         metrics = evaluator.state.metrics
    #         avg_accuracy = metrics['accuracy']
    #         avg_nll = metrics['nll']
    #         tqdm.write(
    #             "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
    #                 .format(engine.state.epoch, avg_accuracy, avg_nll))
    #
    #         pbar.n = pbar.last_print_n = 0
    #
    #     trainer.run(train_loader, max_epochs=epochs)
    #     pbar.close()


    def fitModel(self, batch_size=1024, epochs=500, vsplit=0.1):
        # fit the model
        a = datetime.now()
        self.history = self.model.fit(
            # T_train, rho_train,
            self.X_train, self.y_train,
            epochs=int(epochs),
            batch_size=int(batch_size),
            validation_split=vsplit,
            verbose=2,
            callbacks=self.callbacks_list,
            shuffle=True)
        b = datetime.now()
        c = b-a
        print('The training took: ' + str(int(c.total_seconds()))+' seconds.')


    # def prediction(self):
    #     # loads the best weights
    #     self.model.load_weights("./weights.best.hdf5")
    #
    #     self.predict_y = self.model.predict(self.X_test)


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


    # def plotPredict(self):
    #     #######################
    #     # 1.Plot actual vs prediction for training set
    #     #######################
    #     fig = plt.figure(2)
    #     for prdt, ref in zip(self.rho_predict, self.rho_test):
    #         plt.plot(self.T_scaler.inverse_transform(self.T_test), self.rhoTP_scaler.inverse_transform(prdt), 'b-')
    #         plt.plot(self.T_scaler.inverse_transform(self.T_test), self.rhoTP_scaler.inverse_transform(ref), 'r:')
    #     plt.legend(['predict', 'CoolProp'], loc='upper right')
    #     plt.title(self.test_points)
    #     plt.show(block=False)

    def plotAccuracy(self, target = 'rho', all = False):
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
                plt.title('R2 = ' + str(round(TestR2Value,5)) + ' for: '+tar)

        else:

            target_id = [ind for ind, x in enumerate(column_list) if x == target]
            target_id = target_id[0]

            TestR2Value = metrics.r2_score(self.predict_y[:, target_id], self.y_test[:, target_id])
            print("Test Set R-Square = ", TestR2Value)

            fig = plt.figure(3)
            plt.plot(self.predict_y[:, target_id], self.y_test[:, target_id], 'k^', ms=3, mfc='none')
            plt.title('R2 =' + str(round(TestR2Value,5)) + ' for '+target)

        plt.show(block=False)


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

    def writDNN(self,name=''):
        '''Write a .dnn file to use it with OpenFoam'''
        if name == '':
            print('Give it a name!')
        else:
            cntk.combine(self.model.outputs).save(name)


    # XGBoost classifier!
    def xgboost(self,depth = 7):
        tuned_params = {"objective":"reg:linear",'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': depth}
        self.multiReg = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear', params = tuned_params))
        self.multiReg.fit(self.X_train,self.y_train)
        self.predict_y = self.multiReg.predict(self.X_test)
        score = metrics.r2_score(self.predict_y, self.y_test)
        print('The R2 score from XGBoost is: ',score)


class createDataset(Dataset):

    def __init__(self, X, y):
        self.X = X.astype('float')
        self.y = y.astype('float')

        try:
            assert type(self.X) == np.ndarray
        except AssertionError:
            print('Data has to be numpy.ndarray!')

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self,index):

        # Convert image and label to torch tensors
        # X_tensor = torch.tensor(self.X[index,:],dtype=torch.float32)
        # y_tensor = torch.tensor(self.y[index,:],dtype=torch.float32)

        X_tensor = self.X[index, :]
        y_tensor = self.y[index,:]

        return X_tensor, y_tensor

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.X)


