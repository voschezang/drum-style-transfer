import os, numpy as np, pandas
if __name__ == "__main__":
    np.random.seed(333)
import mido  # , rtmidi, rtmidi_
import matplotlib.pyplot as plt

import config
import setup
import midi
from midi import generators as g
from utils import utils, io, plot

if __name__ == "__main__":
    n = 10
    multiTrack = True
    context, x_train, labels = setup.import_data(
        setup.init(),
        n,
        multiTrack=multiTrack,
        dim4=True,
        dirname='drum_midi',
        r=True)
    config.info('x_train', x_train.shape)

    # plot.single(result[0, :30, :, 0])
