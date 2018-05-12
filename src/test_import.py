import os, numpy as np, pandas
if __name__ == "__main__":
    np.random.seed(333)
import mido  # , rtmidi, rtmidi_
import matplotlib.pyplot as plt

import config
import setup
import models
import midi
from midi import generators as g
from utils import utils, io, plot

if __name__ == "__main__":
    context = setup.init()
    n = 10
    multiTrack = True
    x_train, labels = setup.import_data(
        setup.init(),
        n,
        multiTrack=multiTrack,
        dim4=True,
        dirname='drum_midi',
        r=True)
    config.info('x_train', x_train.shape)

    plot.single(x_train[0, :80, :, 0])
    batch_size = 1
    datagen = models.ImageDataGenerator(x_train, batch_size, 0.01, False)
    x = x_train
    m = 1
    for batch_i, (x_batch, y_batch) in enumerate(
            datagen.flow(x[:m], x[:m], batch_size)):
        x_ = x_batch
        # x_ = datagen.shuffle_3rd_dim(x_)
        x_ = datagen.shuffle_3rd_dim_soft(
            x_, scale=0.5, mutation_rate=0.5, verbose=True)
        break

    plot.single(x_[0, :80, :, 0])
