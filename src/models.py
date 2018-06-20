""" Keras models, using the functional interface

build() creates a VAE, encoder and generator
use vae.load_weights('myfile.h5') to load a pre-trained model
"""
from __future__ import division

import config  # incl. random seed

import numpy as np
import collections
from sklearn.decomposition import PCA
from scipy.stats import norm
import keras
from keras.utils import to_categorical
from keras import optimizers, backend as K
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from utils import utils, plot

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


def gen_latent(generator,
               batch_size=2,
               latent_dim=2,
               x_encoded=0.,
               latent_indices=(0, 1),
               n=10,
               m=4,
               min_x=0.05,
               max_x=0.95,
               min_y=0.05,
               max_y=0.95,
               assume_gaussion=True,
               plot_result=False,
               v=0):
    """ Original: keras.keras.examples.variational_autoencoder
    :x_encoded :: float | [ float ]

    to swap x,y set `latent_indices`` to (1,0)
    """
    if not isinstance(x_encoded, np.ndarray):
        x_encoded = np.repeat(x_encoded, latent_dim)
    if v: print(x_encoded.shape, x_encoded)
    x_decoded = generator.predict(x_encoded.reshape([1, latent_dim]))

    # Coordinate grid
    grid_x = np.linspace(min_x, max_x, n)
    grid_y = np.linspace(min_y, max_y, m)
    # linearly spaced coordinates on the unit square were transformed through
    # the inverse CDF (ppf) of the Gaussian to produce values of the latent
    # variables z, since the prior of the latent space is Gaussian
    if assume_gaussion:
        grid_x = norm.ppf(grid_x)
        grid_y = norm.ppf(grid_y)

    # Generation
    result = collections.defaultdict(dict)
    for yi in grid_x:
        for xi in grid_y:
            z_sample = x_encoded.copy()
            z_sample[np.array(latent_indices)] = (xi, yi)
            # z_sample = np.array([[yi, xi]])
            z_sample = np.tile(z_sample, batch_size).reshape(
                batch_size, latent_dim)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            result[yi][xi] = x_decoded[0]

    if plot_result:
        plot.multi(result, v=v)
    return result


########################################################################
### Model construction
########################################################################
"""
If using a pre-trained model, make sure that the number of latent dims and the number of midi-notes are equal
"""


def build(input_shape=(160, 10, 1), latent_dim=10, epsilon_std=1.):
    timesteps, notes, _ = input_shape
    encoder_model, encoder_input, z_mean, z_log_var = encoder(
        input_shape, latent_dim)
    z_input = encoder_model(encoder_input)

    # this function must be defined locally in order for the model to be serializable
    # using the following inline lambda function will result in an error when .to_json() is called
    #   lambda args: sample(args, z_mean, z_log_var, latent_dim, epsilon_std)
    def sample(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dim),
            mean=0.,
            stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z_output = Lambda(sample)(z_input)
    decoder_model = decoder(z_output, latent_dim, input_shape)

    # VAE
    vae_input = Input(shape=(input_shape, ))
    vae_input = encoder_input
    vae_output = decoder_model(z_output)
    # vae_output = decoder_model(z_mean)
    vae = Model(vae_input, vae_output, name='vae-model-')

    vae_loss_ = vae_loss(
        vae_input,
        vae_output,
        z_mean,
        z_log_var,
        timesteps,
        notes,
        beta=0.75,
        gamma=0.05)
    vae.add_loss(vae_loss_)
    vae.compile(optimizer='adam')

    # Encoder (matrices -> latent vectors)
    encoder_ = Model(encoder_input, z_mean)
    # Generator (latent vectors -> matrices)
    # generator_input = Input((latent_dim, ))
    # generator_layers_ = utils.composition(decoders, generator_input)
    # generator = Model(generator_input, generator_layers_)
    return vae, encoder_, decoder_model


def build_manual(input_shape=(160, 10, 1), latent_dim=10, epsilon_std=1.):
    """
    this model is not serializable due to the lambda layers
    """
    timesteps, notes, _ = input_shape
    encoder_model, encoder_input, z_mean, z_log_var = encoder(
        input_shape, latent_dim)
    sample_ = lambda args: sample(args, z_mean, z_log_var, latent_dim, epsilon_std)
    z_input = encoder_model(encoder_input)
    z_output = Lambda(sample_)(z_input)
    decoders = list_decoders(input_shape)
    decoded = utils.composition(decoders, z_output, verbose=False)

    # VAE: Full model (to train)
    vae_input = encoder_input
    vae_output = decoded
    vae = Model(vae_input, vae_output)
    vae_loss_ = vae_loss(
        vae_input,
        vae_output,
        z_mean,
        z_log_var,
        timesteps,
        notes,
        beta=0.75,
        gamma=0.05)
    vae.add_loss(vae_loss_)
    vae.compile(optimizer='adam')

    # Encoder (matrices -> latent vectors)
    encoder_ = Model(encoder_input, z_mean)

    # Generator (latent vectors -> matrices)
    generator_input = Input((latent_dim, ))
    generator_layers_ = utils.composition(decoders, generator_input)
    generator = Model(generator_input, generator_layers_)

    return vae, encoder_, generator


# internal functions


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
        extra_decoders += [Dense(w, activation='elu')]

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


def decoder(input_, latent_dim=10, output_shape=(40, 10, 1)):
    """ decoder_input = z_output
    decoder.predict :: z -> x
    the decoder models p(x|z), with an assumption over p(z)
    """
    # image_data_format = 'channels_last'
    timesteps, notes, channels = output_shape

    decoder_input = Input(shape=(latent_dim, ))

    h = decoder_input
    h = Dense(256)(h)
    h = LeakyReLU(alpha=0.3)(h)

    # add a bypassed layer
    w = 256
    h_bypass = Dense(w, activation='relu')(h)
    extra_decoders = []
    for _ in range(3):
        h_bypass = Dense(w, activation='elu')(h_bypass)

    h = Add()([h, h_bypass])
    h = BatchNormalization(momentum=0.9)(h)

    n = 10  # 5
    h = RepeatVector(n)(h)
    h = Bidirectional(LSTM(128, return_sequences=True))(h)

    # Embedding decoder

    embedding_len = int(timesteps / n)
    filters = 250
    h = TimeDistributed(Dense(filters, activation='relu'))(h)
    h = TimeDistributed(Dense(notes * embedding_len, activation='sigmoid'))(h)
    h = Reshape((timesteps, notes, 1))(h)

    return Model(decoder_input, h, name='decoder_model-')


def vae_loss(vae_input,
             vae_output,
             z_mean,
             z_log_var,
             timesteps=40,
             notes=9,
             beta=0.5,
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
    # bin. cross-entropy
    xent_loss = timesteps * notes * keras.metrics.binary_crossentropy(
        vae_input_, vae_output_)
    # Kullback-Leibler divergence
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
