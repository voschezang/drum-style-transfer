"""
Convert numpy.ndarray representations of midifiles
"""

from __future__ import division

import numpy as np
import collections
import mido
from typing import List, Dict

import config
import errors
import midi
from midi import pitches, encode
from midi import generators as g
from midi import NoteVector, MultiTrack, Track
from utils import utils


def identity(c, matrix, v=0):
    """decode & encode multiple MidiTracks
    (conversion is lossy)

    matrices :: MultiTrack | [MultiTrack ]
    """
    if len(matrix.shape) == 4:
        return encode.midiFiles(c, tracks(c, matrix, v=v), v=v)
    else:
        return encode.midiFile(c, track(c, matrix, v=v), v=v)


def tracks(c, matrices, v=0) -> List[mido.MidiTrack]:
    """
    matrices :: np.ndarray :: (samples,) + MultiTrack
    """
    return [track(c, matrices[i], v=v) for i in range(matrices.shape[0])]


def track(c, matrix: MultiTrack, transpose=0, name='track_01',
          v=0) -> mido.MidiTrack:
    # c :: data.Context
    # matrix :: [ vector per instance ]
    # vector :: [ notes ]
    if not isinstance(matrix, MultiTrack):
        if not len(matrix.shape) == 3:
            if v:
                config.debug('decode_track - input was not MultiTrack.',
                             'Assuming MultiTrack')

    # decode notes for each instance
    track = mido.MidiTrack()
    track.append(mido.MetaMessage('track_name', name=name))
    # msgs = []
    t = 0
    for i, vector in enumerate(matrix):
        # lookahead_matrix = the part of the matrix that occurred within
        # 'PADDING' cells before 'i'
        if i == 0:
            lookahead_matrix = None
        elif i < midi.PADDING:
            lookahead_matrix = matrix[:i]
        else:
            lookahead_matrix = matrix[i - midi.PADDING:i]
        # msgs :: mido.Message, with absolute t in seconds
        msgs = notes(c, NoteVector(vector), t, lookahead_matrix, transpose)
        track.extend(msgs)
        t += c.dt

    # convert absolute time in seconds to relative ticks
    track.sort(key=lambda msg: msg.time)
    track = midi.convert_time_to_relative_value(
        track, lambda t: midi.second2tick(c, t), v=v)

    mid = mido.MidiFile()
    mid.ticks_per_beat = c.ticks_per_beat
    mid.tracks.append(track)
    if v: config.info('len, max_t', mid.length, c.max_t)
    return mid


def notes(c, notes: NoteVector, t, lookahead_matrix=None, transpose=0,
          v=0) -> List[mido.Message]:
    # :t :: seconds
    # msg.time = absolute, in seconds
    if not isinstance(notes, NoteVector):  # np.generic
        errors.typeError('numpy.ndarray', notes)
    msgs = []
    for note_index, velocity in enumerate(notes):
        if lookahead_matrix is None or lookahead_matrix[:, note_index].max(
        ) < midi.MIDI_NOISE_FLOOR:
            msgs.extend(note(c, note_index, velocity, t, transpose, v=v))
    return msgs


def note(c, note_index, velocity, t, transpose=0, v=0):
    # return ::  [] | a list of midi messages (note on, note off)
    if velocity < midi.MIDI_NOISE_FLOOR:
        return []
    if note_index < midi.SILENT_NOTES:
        return []
    # Convert note_index in array to actual note-value
    # note = midi.LOWEST_NOTE + note_index - midi.SILENT_NOTES
    note = _note(note_index)
    if note > midi.HIGHEST_NOTE:
        if v: config.debug('decode_note: note index > highest note')
    return g.note_on_off(c, note + transpose, 127, t)


def _note(note_index):
    i = note_index - midi.SILENT_NOTES - 1
    return midi.USED_PITCHES[i][0]
