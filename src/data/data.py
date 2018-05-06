""" Functions that are specific to our dataset

Context contains (global) values that are relevant for all midi and other data

"""
import os, pandas, numpy as np, collections
import mido

import config

from utils import utils, io
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


def init():
    print('Setting up params\n')
    max_t = 2.
    dt = 0.02  # T, sampling interval. quantized time, must be > 0
    n_instances = round(max_t / dt)  # vector length
    note_length = 0.03  # seconds
    bpm = 120.  # default bpm
    tempo = mido.bpm2tempo(bpm)
    # ticks_per_beat: 96 | 220 | 480 # midi resolution
    ticks_per_beat = mido.MidiFile().ticks_per_beat
    context = Context(max_t, dt, n_instances, note_length, bpm, tempo,
                      ticks_per_beat)
    print('max min f', utils.max_f(dt), utils.min_f(max_t))
    print(' >>', context)
    return context


def import_data(context, n=2, multiTrack=True, dim4=False):
    # multiTrack = flag to enable matrices with multiple notes (defined in data.midi)
    print('Importing midi-data\n')
    dirname = config.dataset_dir + 'examples/'
    midis, labels = io.import_data(context, dirname, n)

    print('\nEncoding midi-data\n', midis)

    print('> -> multi-track =', multiTrack)
    reduce_dims = True  # rm unused midi-notes
    velocity = 1.
    matrices = midi.encode_midiFiles(
        context, midis, multiTrack, reduce_dims, velocity, dim4=dim4)
    return context, matrices, labels


# TODO omit channel info?
def midi_to_matrix(midi):
    ls = []
    for msg in midi:
        print('is_meta: %s | bytes():' % msg.is_meta, msg.bytes())
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
