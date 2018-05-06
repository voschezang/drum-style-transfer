import keras.backend as K
# from keras import initializers
from keras.layers import Layer

from capsule.capsulefunctions import squash, softmax  #, margin_loss


class Length(Layer):
    # from .capsulelayers from https://github.com/XifengGuo/CapsNet-Keras
    """
    Compute the length of vectors. This is used to compute a Tensor that has
    the same shape with y_true in margin_loss. Using this layer as model's
    output can directly predict labels by using
      `y_pred = np.argmax(model.predict(x), 1)`

    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Capsule(Layer):
    # from https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn_capsule.py
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).

    input shape :: (batch_size, input_num_capsule, input_dim_capsule)
    output shape :: (batch_size, num_capsule, dim_capsule)


    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            shape = (1, input_dim_capsule, self.num_capsule * self.dim_capsule)
            self.kernel = self.add_weight(
                name='capsule_kernel',
                # dense: every input connects to every input of every capsule
                shape=shape,
                initializer='glorot_uniform',
                trainable=True)
            print('shared weights, shape =', shape, np.prod(shape))
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.

        This change can improve the feature representation of Capsule.

        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        # Routing algorithm
        # - what does n_routing (high level)?
        # The prior for coupling coefficient, initialized as zeros. (?)
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            if K.backend() == 'theano':
                outputs = K.sum(outputs, axis=1)
            outputs = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            if i < self.routings - 1:
                # b = K.batch_dot(outputs, hat_inputs, [2, 3])
                b += K.batch_dot(outputs, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    outputs = K.sum(outputs, axis=1)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
