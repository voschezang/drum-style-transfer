""" NN models
"""
from __future__ import division

import config  # incl. random seed

import numpy as np
from sklearn.decomposition import PCA
import keras
from keras.utils import to_categorical
from keras import optimizers, backend as K
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from utils import utils

########################################################################
### Model applications
########################################################################


def transfer_style(encoder, decoder, stylesA, stylesB, samples=[], amt=1.):
    # stylesA, stylesB, samples = lists of samples that can be encoded by `encoder`
    # encoder output must be a np array
    transformation = extract_transformation(encoder, stylesA, stylesB)
    latent_vectors = encoder.predict(samples)
    latent_vectors_ = apply_transformation(latent_vectors, transformation, amt)
    return decoder.predict(latent_vectors_)


def extract_transformation(encoder, stylesA, stylesB) -> np.array:
    # stylesA, stylesB = lists of samples that can be encoded by `encoder`
    # extract the linear latent transformation that corresponds with A -> B
    latent_vectors_A = encoder.predict(stylesA)
    latent_vectors_B = encoder.predict(stylesB)
    return latent_vectors_B.mean(axis=0) - latent_vectors_A.mean(axis=0)


def apply_transformation(vectors: np.array, transformation: np.array,
                         amt=1.) -> np.array:
    # np automatically maps the transformation to every instance with (+)
    return vectors + transformation * amt


########################################################################
### Model construction
########################################################################
"""
# For example

encoder_model, encoder_input, z_mean, z_log_var = encoder(input_shape)
encoder_model.summary()
sample_ = lambda args: models.sample(args, z_mean, z_log_var, latent_dim,
                                                             epsilon_std)
z_input = encoder_model(encoder_input)
z_output = Lambda(sample_)(z_input)
decoders = list_decoders(input_shape)
# VAE model
vae_input = encoder_input
vae_output = decoded
vae = Model(vae_input, vae_output)
vae_loss = vae_loss(beta=0.5)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
# Encoder
encoder = Model(encoder_input, z_mean)
# Generator (decoder without sampling)
generator_input = Input((latent_dim,))
generator_layers_ = utils.composition(decoders, generator_input)
generator = Model(generator_input, generator_layers_)
"""


def sample(args, z_mean, z_log_var, latent_dim=2, epsilon_std=1.):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


def encoder(input_shape, latent_dim=2):
    encoder_input = Input(shape=input_shape)
    nodes = np.prod(input_shape)
    timesteps, notes, channels = input_shape

    # Convolution
    h = encoder_input
    k = (2, 1)
    s = (2, 1)

    h = Reshape((timesteps, notes))(h)
    h = Conv1D(
        64, kernel_size=2, strides=1, activation='relu', padding='valid')(h)

    h = Bidirectional(LSTM(128))(h)

    # Z Mean, Variance
    z_mean = Dense(latent_dim, name='z_mean')(h)  # no activation='relu'
    z_log_var = Dense(latent_dim, name='z_log_var')(h)  # no activation='relu'

    encoder_output = [z_mean, z_log_var]
    encoder_model = Model(encoder_input, encoder_output, name='encoder_model-')
    return encoder_model, encoder_input, z_mean, z_log_var


def list_decoders(output_shape):
    # decoder_input = z_output
    # h = decoder_input
    # :output_shape = (timesteps, channels, channels) || (batches, filters, timesteps, channels)
    # keras offers just Conv2DTranspose and not Conv1DTranspose
    # - use 2D images during upsampling :: (timesteps, notes, channels) => (timesteps, notes, filters)
    # - use 1D images to optimize reconstruction :: (timesteps, filters) => (timesteps, notes)

    # image_data_format = 'channels_last'
    # goal shape: (timesteps, notes, channels)
    # start with the 'reverse': lots of small imgs => few large img

    timesteps, notes, channels = output_shape

    # keras.examples.variational_autoencoder_deconv.py
    decoders = []
    decoders += [Dense(256)]
    decoders += [LeakyReLU(alpha=0.3)]

    # add a bypass layer
    w = 256
    decoders += [Dense(w, activation='relu')]
    extra_decoders = []
    for _ in range(3):
        extra_decoders += [
            Dense(w, activation='elu', bias_initializer='zeros')
        ]

    extra_d = Lambda(lambda layer: utils.composition(extra_decoders, layer))
    decoders += [Lambda(lambda layer: Add()([layer, extra_d(layer)]))]

    decoders += [BatchNormalization(momentum=0.9)]
    n = 10  # 5

    decoders += [RepeatVector(n)]
    decoders += [Bidirectional(LSTM(128, return_sequences=True))]

    # Embedding decoder

    embedding_len = int(timesteps / n)
    filters = 250
    decoders += [TimeDistributed(Dense(filters, activation='relu'))]
    decoders += [
        TimeDistributed(Dense(notes * embedding_len, activation='sigmoid'))
    ]
    decoders += [Reshape((timesteps, notes, 1))]

    return decoders


def vae_loss(vae_input,
             vae_output,
             z_mean,
             z_log_var,
             timesteps=40,
             notes=9,
             beta=1.,
             extra_loss_f=keras.losses.mean_absolute_error,
             gamma=0.5):
    """
    vae_input, vae_output == x == y :: np.ndarray
    z_mean, z_log_var :: np.ndarray
    beta = disentanglement amount
    gamma = multiplier for extra loss function
    """
    vae_input_ = K.flatten(vae_input)
    vae_output_ = K.flatten(vae_output)
    xent_loss = timesteps * notes * keras.metrics.binary_crossentropy(
        vae_input_, vae_output_)
    kl_loss = -1. * K.sum(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # (optional) kl_loss = max(kl_loss, free_bits)
    extra_loss = timesteps * notes * extra_loss_f(vae_input_, vae_output_)

    # change beta
    #     beta = ((1.0 - tf.pow(hparams.beta_rate, tf.to_float(self.global_step)))
    #             * hparams.max_beta)
    #     self.loss = tf.reduce_mean(r_loss) + beta * tf.reduce_mean(kl_cost)
    # y_true, y_pred, z_mean, z_log_var, timesteps=150, notes=3, beta=1.

    vae_loss = K.mean(xent_loss + beta * kl_loss + gamma * extra_loss)
    return vae_loss


class ImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    def __init__(self,
                 x,
                 batch_size=32,
                 phase_mod=0.2,
                 whitening=False,
                 zca_epsilon=1e-6):
        keras.preprocessing.image.ImageDataGenerator.__init__(
            self,
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by dataset std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_epsilon=zca_epsilon,
            zca_whitening=whitening,
            rotation_range=0,  # randomly rotate images in 0 to 180 degrees
            width_shift_range=0.,  # note-channel mod, but not shuffled
            height_shift_range=phase_mod,  # start_t, phase
            horizontal_flip=False,  # reverse
            vertical_flip=False)

        self.x = x
        self.fit(x)
        self.batch_size = batch_size

    def __len__(self):
        # n batches / epoch = #samples / batch_size
        # floor() to prevent occurences of samples multiple times
        return int(np.floor(self.x.shape[0] / self.batch_size))

    def shuffle_3rd_dim(self, x_batch):
        """ Shuffle the 3rd matrix dim with a bias for soft mutations
        :x_batch :: (samples, timesteps, 3rd_dim, channels)
        """
        z_batch = np.empty_like(x_batch)
        for batch_i in range(x_batch.shape[0]):
            indices = np.arange(x_batch.shape[2])
            np.random.shuffle(indices)
            for i, j in enumerate(indices):
                z_batch[batch_i, :, i] = x_batch[batch_i, :, j]

        return z_batch

    def shuffle_3rd_dim_soft(self,
                             x_batch,
                             rate=1,
                             intensity=0.5,
                             scale=0.5,
                             verbose=0):
        """ Shuffle the 3rd matrix dim with a bias for soft mutations
        :x_batch :: (samples, timesteps, 3rd_dim, channels)
        :rate = amount (%) of samples that are mutated
        :intensity = amount of mutations in a sample, relative to len(3rd_dim)
          in range [0, ..]
        :scale = intensity of mutations, relative to len(3rd_dim)
          standard_deviation = scale * len(3rd_dim)
        """
        # z_batch = np.empty_like(x_batch)
        z_batch = x_batch.copy()
        length = x_batch.shape[2]
        indices = np.arange(length)
        std = scale * length
        for batch_i in range(x_batch.shape[0]):
            if np.random.random() < rate:
                if verbose > 1: print('batch_i: %i' % batch_i)
                for i, j in enumerate(
                        self._shuffle_indices(indices, std, verbose)):
                    z_batch[batch_i, :, i] = x_batch[batch_i, :, j]

        return z_batch

    def _shuffle_indices(self, indices, intensity, std=1, verbose=0):
        indices = indices.copy()
        length = indices.shape[0]
        n_iterations = int(round(length * np.random.random()**(1 / intensity)))
        for _ in range(n_iterations):
            i1 = np.random.choice(indices)
            i2 = int((i1 + np.random.normal(0, std)) % length)
            if verbose > 0: print(i1, i2)
            indices[i1], indices[i2] = indices[i2], indices[i1]
        return indices


# class DataGenerator2(keras.utils.Sequence):
#     def __init__(self, x, y=None, batch_size=32, shuffle=True, return_y=False):
#         self.x = x
#         self.y = y if y is not None else x
#         self.return_y = return_y
#         self.n = x.shape[0]
#         self.batch_size = batch_size
#         self.shuffle_samples = shuffle
#         self.shuffle_notes = shuffle
#         self.indices = []
#         self.on_epoch_end()

#     def on_epoch_end(self):
#         self.indices = np.arange(self.n)
#         if self.shuffle_samples:
#             np.random.shuffle(self.indices)

#     def __len__(self):
#         # n batches / epoch = #samples / batch_size
#         # floor() to prevent occurences of samples multiple times
#         return int(np.floor(self.x.shape[0] / self.batch_size))

#     def __data_generation(self, batch_indices):
#         # x_batch :: (samples, timesteps, notes)
#         if batch_indices[-1] > self.x.shape[0]:
#             print('__data_generation - batch_indices too large',
#                   batch_indices[-1], self.x.shape)
#         if not self.shuffle_notes:
#             x_batch = self.x[batch_indices]  # .copy() if mutating
#         else:
#             # gen batch placeholder
#             x_batch = np.empty_like(self.x[batch_indices])
#             for relative_batch_i, batch_i in enumerate(batch_indices):
#                 note_indices = np.arange(self.x.shape[-1])
#                 np.random.shuffle(note_indices)
#                 for note_i, note_j in enumerate(note_indices):
#                     x_batch[relative_batch_i, :, note_i] = self.x[batch_i, :,
#                                                                   note_j]
#         if self.return_y:
#             return x_batch, x_batch
#         return x_batch, None

#     def __getitem__(self, i):
#         # get batch
#         if i >= self.__len__():
#             print('i >= __len__, index should be smaller', i, self.__len__())
#             i -= 1
#         batch_indices = np.arange(i * self.batch_size,
#                                   (i + 1) * self.batch_size) - 1
#         return self.__data_generation(batch_indices)
