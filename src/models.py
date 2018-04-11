""" NN models
"""

# import nn libs
from sklearn.decomposition import PCA
import keras
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras.layers import Conv2DTranspose, Reshape, MaxPooling2D, UpSampling2D
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
    optimizer = Adam(lr=learning_rate)
    # top_k_categorical_accuracy(y_true, y_pred, k=5)
    # https://keras.io/metrics/
    metrics = ['accuracy']  # , 'mean_squared_error']
    model.compile(
        optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

    return model, summary


########################################################################
### Models
########################################################################


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


# TODO not yet, rnn, but just autoencoder + latent space
