# The modules midi.encode, midi.decode depend on the following functions,
# therefore this module cannot import them

# -*- coding: utf-8 -*-
""" Midi Datastructures
Functions that have to do with midi files
Midi is represented either as mido.midi or np.array

The encoding is not lossless: the data is quantized and meta-data is discarded

Encoding & decoding (mido-np conversion)
During encoding, time is automatically converted, dependent on the meta info of the midifile
During decoding, midi-time (PPQ) is set according to the definitions in `context`.
Thus, the midi-time resolution may change during conversion.

Note-off messages are ignored during encoding, for they often are independent of the actual length of a sound. (The length/decay of the sound of the majority of percussion instruments is determined by the instrument itself, and not by the player)

Midi can be represented in numpy ndarrays, either as
  Track :: (timesteps, note)
  MultiTrack :: (timesteps, notes)

note = [0] | [1]
notes = [i] for i in range(n_notes)

"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
import collections
import mido
from typing import List, Dict

import config
import errors
from utils import utils
# from . import encode
# from . import decode

SILENT_NOTES = 0  # 0: no silent notes | int: silent notes
LOWEST_NOTE = 50
HIGHEST_NOTE = 80
N_NOTES = HIGHEST_NOTE - LOWEST_NOTE + SILENT_NOTES
VELOCITY_RANGE = 127
NOTE_OFF = 'note_off'
NOTE_ON = 'note_on'
MIDI_NOISE_FLOOR = 0.5  # real values below this number will be ignored by midi decoders
PADDING = 3  # n array cells after an instance of a note-on msg (should be > 0)
VELOCITY_DECAY = 0.6  # velocity decay for every padded-cell

DTYPE = 'float32'

# 0.5 to be compatible with binary crossentropy

# TODO - Note c Notes, Track c MultiTrack


class Notes(np.ndarray):
    # array :: (notes)
    # array with floats in range [0,1] for every (midi) note
    # to be used as note-on messages at an instance
    # def __new__(cls, array=np.zeros(N_NOTES)):
    # note: function default args are evaluated once, before runtime
    def __new__(cls, array=None):
        if array is None:
            array = np.zeros(N_NOTES)
        return array.astype(DTYPE).view(cls)


class Note(np.ndarray):
    # array :: [0] | [1]
    # array with a single float in range [0,1]
    # to be used as a note-on message at an instance
    # note: function default args are evaluated once, before runtime
    def __new__(cls, array=None):
        if array is None:
            array = np.zeros(1, dtype=DTYPE)
        return array.astype(DTYPE).view(cls)


class MultiTrack(np.ndarray):
    # array :: (timesteps, Notes)
    # array of Notes, with length 'track-length'
    def __new__(cls, length, dt):
        arr = np.stack([Notes() for _ in range(length)])
        # at every timestep, fill notes with index in range 0:SILENT_NOTES with 1
        arr[:, :SILENT_NOTES] = 1.
        return arr.astype(DTYPE).view(cls)

    def __init__(self, length=100, dt=0.01):
        self.dt = dt  # seconds

    def length_in_seconds(self):
        # n instances * dt, in seconds
        return self.shape[0] * self.dt


class Track(np.ndarray):
    # array :: (timesteps, Note)
    def __new__(cls, array):
        if len(array.shape) == 1:
            # transform a list of float to a list of Note
            return np.expand_dims(array, axis=1).view(cls)
        return array.astype(DTYPE).view(cls)

    def length_in_seconds(self):
        # n instances * dt, in seconds
        return self.shape[0] * self.dt


def multiTrack_to_list_of_Track(matrix: MultiTrack):
    # def split_tracks(matrix: MultiTrack):
    # :: MultiTrack -> list NoteList
    # matrix = array (timesteps, notes)
    tracks = []
    note_indices = matrix.shape[1]
    for i in range(note_indices):
        # ignore notes indices that are not present
        if matrix[:, i].max() > MIDI_NOISE_FLOOR:
            tracks.append(Track(matrix[:, i]))
    return np.stack(tracks)


import numpy as np, collections
import mido
from typing import List, Dict

import config, errors
from utils import utils


def gen_note_on_off(c, note, velocity, t):
    # :t :: seconds
    # return ::  [] | a list of midi messages (note on, note off)
    # velocity *= RANGE
    msg1 = mido.Message(NOTE_ON, note=note, velocity=127, time=t)
    msg2 = mido.Message(
        NOTE_OFF, note=note, velocity=velocity, time=t + c.note_length)
    return [msg1, msg2]


def second2tick(c, t):
    return round(mido.second2tick(t, c.ticks_per_beat, c.tempo))


def combine_notes(v1, v2):
    # v = Notes((v1 + v2).clip(0, 1))
    v = np.maximum(v1, v2)

    if SILENT_NOTES > 0:
        # use a placeholder note to indicate the absence of a note_on msg
        # if for v1 & v2, no notes (except a SILENT_NOTE) are 1,
        #   SILENT_NOTE must be 1 else 0
        if v1[SILENT_NOTES:].max(
        ) < MIDI_NOISE_FLOOR and v2[SILENT_NOTES:].max() < MIDI_NOISE_FLOOR:
            v[:SILENT_NOTES] = 1.
        else:
            v[:SILENT_NOTES] = 0
    return v


def convert_time_to_relative_value(ls, convert_time):
    # convert in place
    current_t = 0
    prev_t = 0
    for msg in ls:
        old_t = msg.time
        if prev_t > old_t:
            config.debug('prev-t >', prev_t, old_t)
        prev_t = old_t
        if old_t < current_t:
            config.debug('old current', old_t, current_t)
        dt = old_t - current_t
        msg.time = convert_time(dt)
        current_t += dt
    return ls


def to_midi(arr):
    # arr :: np.array
    pass


def _normalize_bytes(arr):
    # arr :: np.array
    # TODO check max mido midi value
    return arr / 256.


def _denormalize(arr):
    # arr :: np.array
    return arr * 256.


# def reduce_multiTrack_dims(matrix):
#     if reduce_dims:
#         matrix = reduce_dims(matrix)
#         indices = []
#         for note_i in np.arange(matrix.shape[-1]):
#             if tracks[:, :, note_i].max() > MIDI_NOISE_FLOOR:
#                 indices.append(note_i)
#         tracks = tracks[:, :, indices]
#         config.info('reduced dims:', tracks.shape)

# def reduce_matrix_list_dims(matrix):
