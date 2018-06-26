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
from __future__ import division

import numpy as np, bz2, collections
import mido
from typing import List, Dict

import config
import errors
from utils import utils
from midi import pitches

KIT_SIZE = 1  # 1 or more of each instrument
USED_DRUMS = [note_list[:KIT_SIZE]
              for note_list in pitches.DRUMS]  # = [BD, SN, HH]
# USED_PITCHES = np.concatenate(USED_DRUMS)
# USED_PITCHES = [[BD],[SN1], [SN2,SN3]]
# the index of a value corresponds to the pitch
# USED_PITCHES = [[[note_list[i]]
#                  for i in range(KIT_SIZE - 1)] + [note_list[KIT_SIZE:]]
#                 for note_list in pitches.DRUMS]
USED_PITCHES = pitches.used_note_list(pitches.DRUMS, KIT_SIZE)

SILENT_NOTES = 0  # 0: no silent notes | int: silent notes
UNKNOWN_NOTES = 1  # must be 1
LOWEST_NOTE = min(min(ls) for ls in USED_PITCHES)
HIGHEST_NOTE = max(max(ls) for ls in USED_PITCHES)

# LOWEST_NOTE = min(USED_PITCHES)
# HIGHEST_NOTE = max(
#     USED_PITCHES) + 1  # the highest note indicates unknown pitches
# N_NOTES = HIGHEST_NOTE - LOWEST_NOTE + SILENT_NOTES
N_NOTES = SILENT_NOTES + len(USED_PITCHES) + UNKNOWN_NOTES
VELOCITY_RANGE = 127
NOTE_OFF = 'note_off'
NOTE_ON = 'note_on'
MIDI_NOISE_FLOOR = 0.1  # values below this number will be ignored by midi decoders
PADDING = 3  # n array cells after an instance of a note-on msg (should be > 0)
VELOCITY_DECAY = 0.2  # velocity decay for every padded-cell

DTYPE = 'float32'


class ReduceDimsOptions:
    GLOBAL = 'global'
    MIDIFILE = 'MidiFile'
    NONE = 'none'


class NoteVector(np.ndarray):
    """ Array with floats in range [0,1] for every (midi) note
    to be used as note-on messages at an instance
    """

    # def __new__(cls, array=np.zeros(N_NOTES)):
    # note: function default args are evaluated once, before runtime

    def __new__(cls, array=None, n_notes=None):
        # array :: [ notes ] | None
        if array is None:
            if n_notes is None:
                n_notes = N_NOTES
            array = np.zeros(N_NOTES)
        return array.astype(DTYPE).view(cls)


class MultiTrack(np.ndarray):
    """ np.ndarray :: (timesteps, NoteVector),
    with length 'track-length'

    """

    def __new__(cls, n_timesteps, n_notes=None):
        if n_notes is None:
            n_notes = N_NOTES
        arr = np.zeros([n_timesteps, n_notes], dtype=DTYPE)
        # at every timestep, fill notes with index in range 0:SILENT_NOTES with 1
        arr[:, :SILENT_NOTES] = 1.
        return arr.astype(DTYPE).view(cls)

    def from_array(arr):
        m = MultiTrack(arr.shape[0], arr.shape[1])
        if len(arr.shape) == 2:
            m[:, :] = arr[:, :]
        elif len(arr.shape) == 3:
            m[:, :] = arr[:, :, 0]
        else:
            raise TypeError('array should have shape (timesteps, notes, )')
        return m

    def length_in_seconds(self):
        # n instances * dt, in seconds
        return self.n_timesteps * self.dt

    def n_timesteps(self):
        return self.shape[0]

    def n_notes(self):
        return self.shape[1]

    def multiTrack_to_list_of_Track(self):
        # return :: [ Track ]
        # matrix = array (timesteps, notes)
        tracks = []
        note_indices = self.shape[1]

        for i in range(note_indices):
            # ignore notes indices that are not present
            if self[:, i].max() > MIDI_NOISE_FLOOR:
                tracks.append(Track(self[:, i]))

        return np.stack(tracks)

    def reduce_dims(self):
        # return :: MultiTrack
        # discard note order
        # TODO parallelize
        used_indices = []
        for note_i in range(self.shape[1]):
            if self[:, note_i].max() > MIDI_NOISE_FLOOR:
                used_indices.append(note_i)
        return self[:, used_indices]

    def fit_dimensions(self, n_timesteps, n_notes):
        # increase dimensions to (n_timesteps, n_notes)
        if self.n_timesteps() < n_timesteps or self.n_notes() < n_notes:
            track = MultiTrack(n_timesteps, n_notes)
            track[:self.n_timesteps(), :self.n_notes()] = self
            return track
        return self


class Track(MultiTrack):
    """ A MultiTrack where NoteVector.length of 1
    """

    def __new__(cls, array):
        return MultiTrack.__new__(array.shape[0], n_notes=1)

    def __init__(self, array):
        # MultiTrack.__init__(array.shape[0], n_notes=1)
        if len(array.shape) == 1:
            # transform [0, 1, ..] => [[0], [1], ..]
            array = np.expand_dims(array, axis=1)
        self[:, 0] = array[:, 0]


def second2tick(c, t):
    return round(mido.second2tick(t, c.ticks_per_beat, c.tempo))


def concatenate(multiTracks: List[MultiTrack]) -> MultiTrack:
    return np.concatenate(multiTracks, axis=0)


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


def convert_time_to_relative_value(ls, convert_time, v=0):
    # convert in place
    current_t = 0
    prev_t = 0
    for msg in ls:
        old_t = msg.time
        if prev_t > old_t:
            if v: config.debug('prev-t >', prev_t, old_t)
        prev_t = old_t
        if old_t < current_t:
            if v: config.debug('old current', old_t, current_t)
        dt = old_t - current_t
        msg.time = convert_time(dt)
        current_t += dt
    return ls


def reduce_MultiTrack_list_dims(tracks, v=0):
    # [ MultiTrack ] -> [ MultiTrack ]
    used_indices = []
    for note_i in range(tracks.shape[-1]):
        if tracks[:, :, note_i].max() > MIDI_NOISE_FLOOR:
            used_indices.append(note_i)
    tracks = tracks[:, :, used_indices]
    if v: config.info('reduced mt list dims:', tracks.shape)
    return tracks  # return tracks[:, :, indices]


def is_note_on(msg: mido.Message):
    return not msg.is_meta and msg.type == NOTE_ON
