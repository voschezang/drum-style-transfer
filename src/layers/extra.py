import keras.layers


class Resample(keras.layers.Layer):
    def __new__(self, input_shape, length):
        timesteps, notes, channels = input_shape
        n = int(timesteps / length)
        # TODO crop input if incompatible dims
        # crop((0, timesteps % length))
        return keras.layers.Reshape((n, length, notes, channels))
