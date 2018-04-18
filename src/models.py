""" NN models
"""
import config  # incl. random seed
import numpy as np
# import nn libs
from sklearn.decomposition import PCA
import keras
from keras.utils import to_categorical
from keras import optimizers
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras.layers import Conv2DTranspose, Reshape, MaxPooling2D, UpSampling2D
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import LocallyConnected1D, LocallyConnected2D
from keras.models import Model

########################################################################
### Init
########################################################################


def init(x_train, y_train):
    n_samples = x_train[0]
    input_shape = x_train.shape[1:]  # shape of a single sample
    output_length = y_train.shape[1]  # length of an individual label

    dropout = 0.
    model, summary = model1(input_shape, output_length, dropout)

    learning_rate = 0.01
    # sgd = Keras.optimizers.SGD(lr=0.01, clipnorm=1.)
    optimizer = optimizers.Adam(lr=learning_rate)
    # top_k_categorical_accuracy(y_true, y_pred, k=5)
    # https://keras.io/metrics/
    metrics = ['accuracy']  # , 'mean_squared_error']
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

    return model, summary


########################################################################
### Models
########################################################################

# functional syntax: lambda x: lambda y: z


def resolution_reducer(input_shape, amt=2):
    input_layer = Input(shape=input_shape)
    x = input_layer
    x = MaxPooling1D(pool_size=amt, strides=amt)(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


def model1(input_shape, output_length, dropout=0.10):
    input_layer = Input(shape=input_shape)
    x = input_layer
    x = Flatten()(x)
    #     b = Dense(100, activation='relu')(b)
    #     a = Conv2D(3, kernel_size=(1, 1), activation='relu', input_shape=input_shape)
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    #     model.add(Dropout(dropout))
    return model, model.summary


def encoder(input_shape, output_length, dropout=0.10):
    input_layer = Input(shape=input_shape)
    x = input_layer
    x = Flatten()(x)
    #     x = Dense(10, activation='relu')(x)
    x = Dense(output_length, activation='sigmoid')(x)  # softmax
    model = Model(inputs=input_layer, outputs=x)
    #     model.add(Dropout(dropout))
    return model, model.summary


def decoder(input_length, output_shape, dropout=0.10):
    input_layer = Input(shape=(input_length, ))
    x = input_layer
    shape = (10, 100)  # 1 additional dimension
    x = Dense(np.prod(shape), activation='relu')(x)  # 4*4*8 = 128
    x = Reshape(shape)(x)
    x = UpSampling1D(10)(x)
    x = Conv1D(100, 2, strides=2, activation='relu')(x)  # 50,100
    x = UpSampling1D(output_shape[0] / 50)(x)
    x = Dense(output_shape[1], activation='relu')(x)
    x = Dense(output_shape[1], activation='relu')(x)
    x = Dense(output_shape[1], activation='relu')(x)
    #     x = LocallyConnected1D(output_shape[1], kernel_size=1, activation='relu')(x)
    # x = Dense(output_length, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    #     model.add(Dropout(dropout))
    return model, model.summary


def autoencoder(input_shape,
                output_shape,
                hidden_layer_length=10,
                dropout=0.10,
                verbose=False):
    encode, summary = encoder(input_shape, hidden_layer_length, dropout)
    if verbose:
        summary()
    decode, summary = decoder(hidden_layer_length, output_shape, dropout)
    if verbose:
        summary()
    input_ = Input(shape=input_shape)
    model = Model(input_, decode(encode(input_)))
    return encode, decode, model, model.summary


# TODO not yet, rnn, but just autoencoder + latent space
