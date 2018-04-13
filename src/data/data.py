""" Functions that are specific to our dataset

Context contains (global) values that are relevant for all midi and other data

"""
import os, pandas, numpy as np, collections
import mido

import config

from utils import io
from data import midi
# from utils import utils

Context = collections.namedtuple('Context', [
    'max_t',
    'dt',
    'n_instances',
    'note_length',
    'bpm',
    'tempo',
    'ticks_per_beat',
])

print(""" Context :: namedtuple(
[ max_t = float
, dt = float
, n_instances = int
, note_length = int
, bpm = float
, tempo = float
, ticks_per_beat = int
]
""")


def init(n: int = 2):
    print('Setting up params\n')
    max_t: float = 10.
    dt = 0.01  # quantized time, must be > 0
    n_instances = round(max_t / dt)  # vector length
    note_length = 0.03  # seconds
    bpm = 120.  # bpm
    tempo = mido.bpm2tempo(bpm)
    ticks_per_beat = 96  # 480 # midi resolution
    context = Context(max_t, dt, n_instances, note_length, bpm, tempo,
                      ticks_per_beat)
    print(' >>', context)

    print('Importing midi-data\n')
    dirname = config.dataset_dir + 'examples/'
    midis = io.import_data(context, dirname, n)

    print('\nEncoding midi-data\n', midis)
    arrays = [midi.encode(context, m) for m in midis]
    x_train = np.stack(arrays)
    return context, x_train


# TODO omit channel info?
def midi_to_matrix(midi):
    ls = []
    for msg in midi:
        print('is_meta: %s | bytes():' % msg.is_meta, msg.bytes())
        print(msg)
        if not msg.is_meta:
            ls.append(msg.bytes())
    return np.array(ls)


# def make_compatible(arrays):
#     # :arrays :: list np.array(n,3)
#     smallest = arrays[0].shape[0]
#     for a in arrays:
#         # TODO check dimension priority
#         if a.shape[0] < smallest:
#             smallest = a.shape[0]
#     return [a[0:smallest] for a in arrays]
