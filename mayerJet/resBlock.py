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
