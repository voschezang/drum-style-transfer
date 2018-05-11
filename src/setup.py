# rename data.midi => midi.midi
# rename data.midi_generators => midi.generators
""" Functions that are specific to our dataset

Context contains (global) values that are relevant for all midi and other data

"""
from __future__ import division

import numpy as np
import collections
import mido

import config
import midi
# from midi import encode
# from midi import decode
from utils import utils
from utils import io

Context = collections.namedtuple('Context', [
    'max_t',
    'dt',
    'n_timesteps',
    'note_length',
    'bpm',
    'tempo',
    'ticks_per_beat',
])

print(""" Context :: namedtuple(
[ max_t = float
, dt = float
, n_timestesp = int
, note_length = int
, bpm = float
, tempo = float
, ticks_per_beat = int
]
""")


def init():
    # return Context
    print('Setting up params\n')
    bpm = 120.  # default bpm
    max_t = 2.
    max_bars = 2
    max_t = 60 / bpm * 2 * max_bars
    dt = 0.02  # T, sampling interval. quantized time, must be > 0
    n_timesteps = round(max_t / dt)  # vector length
    note_length = 0.03  # seconds
    tempo = mido.bpm2tempo(bpm)
    # ticks_per_beat: 96 | 220 | 480 # midi resolution
    ticks_per_beat = mido.MidiFile().ticks_per_beat
    context = Context(max_t, dt, n_timesteps, note_length, bpm, tempo,
                      ticks_per_beat)
    print('max min f', utils.max_f(dt), utils.min_f(max_t))
    print(' >>', context)
    print(' sample length: % f' % (max_t / dt))
    print(' max_f: %f, min_f: %f' % (utils.max_f(dt), utils.min_f(max_t)))
    return context


def import_data(context,
                n=2,
                multiTrack=True,
                dim4=True,
                reduce_dims='global',
                dirname='examples',
                r=False,
                velocity=1.):
    # multiTrack = flag to enable matrices with multiple notes (defined in midi.init)
    print('\nImporting midi-data')
    dirname = config.dataset_dir + dirname + '/'
    midis, labels = io.import_mididata(context, dirname, n, r)

    print('\nEncoding midi-data\n', len(midis))

    print('> -> multi-track =', multiTrack)
    matrices = midi.encode.midiFiles(context, midis, multiTrack, reduce_dims,
                                     velocity, dim4)
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
