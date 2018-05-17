from keras import backend as K


def kl_loss(z_mean, z_log_var):
    # z_mean :: (batch_size, latent_dim)
    # z_log_var :: (batch_size, latent_dim)
    return -0.5 * K.sum(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # return - 1. * keras.losses.kullback_leibler_divergence
