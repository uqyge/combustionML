import numpy as np
from sklearn import preprocessing

import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Activation, Input, BatchNormalization, Dropout
from keras import layers
from keras.callbacks import ModelCheckpoint


def res_block(input_tensor, n_neuron, stage, block, bn=False):
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


def label_TP_gen(label, x, fluid):
    out = CP.PropsSI(label, 'T', x[0], 'P', x[1], fluid)
    return out


######################
print('Generate data ...')
# n_train = 20000
nT = 100
nP = 100
n_train = nT * nP

T_min = 0.7
T_max = 1.3

P_min = 1
P_max = 2

fluid = 'nitrogen'

# get critical pressure & temperature
p_c = CP.PropsSI(fluid, 'pcrit')
t_c = CP.PropsSI(fluid, 'Tcrit')

p_vec = np.linspace(P_min, P_max, nP) * p_c
T_vec = np.linspace(T_min, T_max, nT) * t_c

# prepare input
# rho = f(T, P)
# 1. uniform random
T_P_train = np.random.rand(n_train, 2)
# 2. family curves
# T_P_train = np.random.rand(n_train, 1)
# tmp = np.ones((nT, nP))* np.linspace(0, 1, nP)
# T_P_train = np.append(T_P_train, tmp.reshape(-1, 1), axis=1)

train_dict = {0: 'D',
              1: 'V',
              2: 'L',
              3: 'H'}
scalers = []
labels = []
for i in train_dict:
    out_TP_train = np.asarray([label_TP_gen(train_dict[i], x, fluid) for x in (
        T_P_train * [(T_max - T_min) * t_c, (P_max - P_min) * p_c] + [T_min * t_c, P_min * p_c])])

    # normalize train data
    scaler = preprocessing.MinMaxScaler()
    out_TP_train = scaler.fit_transform(out_TP_train.reshape(-1, 1))
    scalers.append(scaler)
    labels.append(out_TP_train)
    # label_TP_train = np.append(label_TP_train, out_TP_train, axis=1)



label_TP_train = np.asarray(labels[0])
for x in labels:
    if not (x == label_TP_train).all():
        label_TP_train = np.concatenate((label_TP_train, x), axis=1)

# normalize test data
T_scaler = preprocessing.MinMaxScaler()
T_test = T_scaler.fit_transform(T_vec.reshape(-1, 1))

######################
print('set up ANN')
# ANN parameters
dim_input = 2
dim_label = 4
n_neuron = 100
batch_size = 1024
epochs = 4000
vsplit = 0.1
batch_norm = False

# This returns a tensor
inputs = Input(shape=(dim_input,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(n_neuron, activation='relu')(inputs)

# less then 2 res_block, there will be variance
x = res_block(x, n_neuron, stage=1, block='a', bn=batch_norm)
x = res_block(x, n_neuron, stage=1, block='b', bn=batch_norm)

# x = res_block(x, n_neuron, stage=1, block='c', bn=batch_norm)
# x = res_block(x, n_neuron, stage=1, block='d', bn=batch_norm)

predictions = Dense(dim_label, activation='linear')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# checkpoint (save the best model based validate loss)
filepath = "./tmp/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min',
                             period=10)
callbacks_list = [checkpoint]

# fit the model
history = model.fit(
    # T_train, rho_train,
    T_P_train, label_TP_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=vsplit,
    verbose=2,
    callbacks=callbacks_list,
    shuffle=True)

# loss
fig = plt.figure()
plt.semilogy(history.history['loss'])
if vsplit:
    plt.semilogy(history.history['val_loss'])
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

# T_P_test = np.append(T_test, np.ones((len(T_test), 1)) * (1.1 - P_min) / (P_max - P_min), axis=1)
# predict = model.predict(T_P_test)

# ######################
print('post processing')
print('model predict')
model.load_weights("./tmp/weights.best.hdf5")

label_dict = {0: ('D', scalers[0]),
              1: ('V', scalers[1]),
              2: ('L', scalers[2]),
              3: ('H', scalers[3])}

test_points = [1.02, 1.05, 1.1, 1.2, 1.5]

for i in label_dict:
    label_predict = []
    label_test = []
    for x_p in test_points:
        T_P_test = np.append(T_test, np.ones((len(T_test), 1)) * (x_p - P_min) / (P_max - P_min), axis=1)
        predict = model.predict(T_P_test)[:, i]
        label_ref = np.asarray([CP.PropsSI(label_dict[i][0], 'T', x_T, 'P', x_p * p_c, fluid) for x_T in T_vec])
        label_predict.append(predict.reshape(-1, 1))
        label_test.append(label_ref)

    # 1.Plot actual vs prediction for training set
    plt.figure()
    for prdt, ref in zip(label_predict, label_test):
        plt.plot(T_scaler.inverse_transform(T_test), label_dict[i][1].inverse_transform(prdt), 'b-')
        plt.plot(T_scaler.inverse_transform(T_test), ref, 'r:')
    plt.legend(['predict', 'CoolProp'], loc='upper right')
    plt.title(label_dict[i][0])
# # 2.L2 accuracy plot
# fig = plt.figure()
# a = np.asarray(label_predict).reshape(-1, 1)
# b = np.asarray(label_test).reshape(-1, 1)
# plt.plot(a, b, 'k^', ms=3, mfc='none')
# # Compute R-Square value for training set
# from sklearn.metrics import r2_score
#
# TestR2Value = r2_score(a, b)
# print("Training Set R-Square=", TestR2Value)
