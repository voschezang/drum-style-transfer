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
from utils import string

Context = collections.namedtuple('Context', [
    'max_t',
    'dt',
    'n_timesteps',
    'note_length',
    'bpm',
    'tempo',
    'ticks_per_beat',
])

Context_desciption = """ Context :: namedtuple(
[ max_t = float
, dt = float
, n_timestesp = int
, note_length = int
, bpm = float
, tempo = float
, ticks_per_beat = int
]
"""


def init():
    print(Context_desciption)
    print('Setting up params\n')
    bpm = 120.  # default bpm
    max_t = 2.
    max_bars = 2
    max_t = 60 / bpm * 2 * max_bars
    dt = 0.05  # T, sampling interval. quantized time, must be > 0
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
                reduce_dims=midi.ReduceDimsOptions.GLOBAL,
                dim4=True,
                dirname='examples',
                r=False,
                velocity=None):
    # multiTrack = flag to enable matrices with multiple notes (defined in midi.init)
    print('\nImporting midi-data')
    dirname = config.dataset_dir + dirname + '/'
    cond = string.is_midifile
    cond = string.is_drumrythm
    midis, labels = io.import_mididata(context, dirname, n, cond, r=r)

    print('\nEncoding midi-data\n', len(midis))

    print('> -> multi-track =', multiTrack, reduce_dims)
    matrices = midi.encode.midiFiles(context, midis, multiTrack, reduce_dims,
                                     velocity, dim4)
    return matrices, labels
