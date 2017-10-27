import numpy as np
import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras import layers

from keras.callbacks import ModelCheckpoint

def res_block(input_tensor, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Dense(100, name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)

    x = Dense(100, name=conv_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x

# vector length
nT = 200000
nP = 100
T_min = 100
T_max = 160

# ANN parameters
dim = 1
batch_size = 1024
epochs = 3000
vsplit = 0.01

fluid = 'nitrogen'

# get critical pressure
p_c = CP.PropsSI(fluid,'pcrit')

p_vec = np.linspace(1,3,nP)
p_vec = (p_vec)*p_c

T_vec = np.zeros((nT))
T_vec[:] = np.linspace(T_min, T_max, nT)

rho_vec = np.zeros((nT))

print('Generate data ...')
for i in range(0,nT):
    rho_vec[i] = CP.PropsSI('D','T',T_vec[i],'P',1.1*p_c,fluid)

# normalize
T_max = max(T_vec)
rho_max = max(rho_vec)

T_norm = T_vec/T_max
rho_norm = rho_vec/rho_max


print('set up ANN')
######################
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=1, init='uniform'))
model.add(Dense(100, activation='relu', init='uniform'))
model.add(Dense(100, activation='relu', init='uniform'))
model.add(Dense(100, activation='relu', init='uniform'))
model.add(Dense(output_dim=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

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
history = model.fit(T_norm,rho_norm,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=vsplit,
                    verbose=2,
                    callbacks=callbacks_list,
                    shuffle=True)


model.load_weights("./tmp/weights.best.hdf5")
predict = model.predict(T_norm)

plt.plot(predict)
plt.plot(rho_norm)
plt.show()

fig = plt.figure()
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
