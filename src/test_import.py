import os, numpy as np, pandas
if __name__ == "__main__":
    np.random.seed(333)
    os.chdir('src')
import mido  # , rtmidi, rtmidi_
import matplotlib.pyplot as plt

# local libs
import config
from data import data, midi, midi_generators as g
from utils import utils, io, plot

if __name__ == "__main__":
    n = 10
    multiTrack = True
    context, x_train, labels = data.import_data(
        data.init(),
        n,
        multiTrack=multiTrack,
        dim4=True,
        dirname='drum_midi',
        r=True)
    config.info('x_train', x_train.shape)

    # plot.single(result[0, :30, :, 0])
