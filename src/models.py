""" NN models
"""
import config  # incl. random seed
import numpy as np
# import nn libs
from sklearn.decomposition import PCA
import keras
from keras.utils import to_categorical
from keras import optimizers, backend as K
from keras.layers import Input, Dense, Activation, Conv2D, Dropout, Flatten
from keras.layers import Conv2DTranspose, Reshape, MaxPooling2D, UpSampling2D
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import LocallyConnected1D, LocallyConnected2D
from keras.models import Model

########################################################################
### Functions
########################################################################


def sample(args, z_mean, z_log_var, latent_dim, epsilon_std):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


# def init(x_train, y_train):
#     n_samples = x_train[0]
#     input_shape = x_train.shape[1:]  # shape of a single sample
#     output_length = y_train.shape[1]  # length of an individual label

#     dropout = 0.
#     model, summary = model1(input_shape, output_length, dropout)

#     learning_rate = 0.01
#     # sgd = Keras.optimizers.SGD(lr=0.01, clipnorm=1.)
#     optimizer = optimizers.Adam(lr=learning_rate)
#     # top_k_categorical_accuracy(y_true, y_pred, k=5)
#     # https://keras.io/metrics/
#     metrics = ['accuracy']  # , 'mean_squared_error']
#     model.compile(
#         optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)

#     return model, summary

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


def resolution_reducer(input_shape, amt=2):
    input_layer = Input(shape=input_shape)
    x = input_layer
    x = MaxPooling1D(pool_size=amt, strides=amt)(x)
    model = Model(inputs=input_layer, outputs=x)
    return model


"""
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
"""
"""
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
"""
"""
def decode_sequence(input_seq, encoder_model, decoder_model):
    max_decoder_seq_length = 500
    
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        print(output_tokens.shape)

        output_ = output_tokens[0, -1, :] # identity in case of 1 batch?
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        # sampled_char = sampled_token_index # reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(output_)

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sentence) >= max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # target_seq[0, 0, :] = output_
        target_seq[0, 0, :] = output_


        # Update states
        states_value = [h, c]

    return np.stack(decoded_sentence)

"""
