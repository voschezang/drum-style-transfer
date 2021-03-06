import numpy as np
from keras.models import model_from_json

import config


def save_model(model, model_name='model'):
    # serialize model to JSON
    model_json = model.to_json()
    fn1 = save_weights(model, model_name)
    fn2 = model_name + '.json'
    with open(fn2, "w") as json_file:
        json_file.write(model_json)
    config.info("Saved model to disk")
    return fn1, fn2


def save_weights(model, model_name='vae'):
    # serialize weights to HDF5
    fn = model_name + '.h5'
    model.save_weights(fn, "w")
    return fn


def load_model(filename):
    with open(filename + '.json', 'r') as json:  # cnn_transfer_augm
        loaded_model_json = json.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename + '.h5')
    config.info("Loaded model from disk")

    # reset seed ?
    np.random.seed(config.seed)
    loaded_model.compile(
        loss="binary_crossentropy", optimizer='adadelta', metrics=['accuracy'])
    config.info('compiled model')
    return loaded_model
