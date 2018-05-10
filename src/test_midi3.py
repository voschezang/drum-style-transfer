import os, numpy as np, pandas
if __name__ == "__main__":
    np.random.seed(333)
    os.chdir('src')
import mido

# local libs
import config
from data import data, midi, midi_generators as g
from utils import utils, io

if __name__ == "__main__":

    n = 100
    multiTrack = False
    context, x_train, labels = data.import_data(
        data.init(), n, multiTrack=multiTrack)
    config.info('x train', x_train.shape)

    print(labels[-2:])

    dn = config.export_dir
    mid_new = midi.decode_track(context, x_train[-2])
    io.export_midifile(mid_new, dn + 'encode_decoder.mid')
