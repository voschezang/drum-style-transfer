import os, numpy as np, pandas
if __name__ == "__main__":
    np.random.seed(333)
import mido  # , rtmidi, rtmidi_
import matplotlib.pyplot as plt

import config
import setup
import models
import midi
import midi.decode
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

    # plot.single(x_train[0, :80, :, 0])
    batch_size = 3
    datagen = models.ImageDataGenerator(x_train, batch_size, 0.01, False)
    x = x_train
    m = 3
    for batch_i, (x_batch, y_batch) in enumerate(
            datagen.flow(x[:m], x[:m], batch_size)):
        x_ = x_batch
        # x_ = datagen.shuffle_3rd_dim(x_)
        x_ = datagen.shuffle_3rd_dim_soft(x_, scale=0.5, verbose=True)
        break

    # plot.single(x_[0, :80, :, 0])

    print('\ntest decoder\n')
    for i in range(2):
        a = midi.MultiTrack.from_array(x_[i])
        b = midi.decode.track(context, a)
        print(b.tracks[0])
        for x in b:
            print(x)
        io.export_midifile(b, config.export_dir + 'y_' + str(i))
