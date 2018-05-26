from __future__ import division

import numpy as np
import collections
import mido
from typing import List, Dict

import config
import errors
import midi
from midi import pitches
from midi import generators as g
from midi import NoteVector, MultiTrack, Track
from utils import utils


def track(c, matrix: MultiTrack, transpose=12) -> mido.MidiTrack:
    # c :: data.Context
    # matrix :: [ vector per instance ]
    # vector :: [ notes ]
    if not isinstance(matrix, MultiTrack):
        if not len(matrix.shape) == 3:
            config.debug('decode_track - input was not MultiTrack.',
                         'Assuming MultiTrack')

    # decode notes for each instance
    track = mido.MidiTrack()
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
        track, lambda t: midi.second2tick(c, t))

    mid = mido.MidiFile()
    mid.ticks_per_beat = c.ticks_per_beat
    mid.tracks.append(track)
    config.info('len, max_t', mid.length, c.max_t)
    return mid


def notes(c, notes: NoteVector, t, lookahead_matrix=None,
          transpose=0) -> List[mido.Message]:
    # :t :: seconds
    # msg.time = absolute, in seconds
    if not isinstance(notes, NoteVector):  # np.generic
        errors.typeError('numpy.ndarray', notes)
    msgs = []
    for note_index, velocity in enumerate(notes):
        if lookahead_matrix is None or lookahead_matrix[:, note_index].max(
        ) < midi.MIDI_NOISE_FLOOR:
            msgs.extend(note(c, note_index, velocity, t, transpose))
    return msgs


def note(c, note_index, velocity, t, transpose=0):
    # return ::  [] | a list of midi messages (note on, note off)
    if velocity < midi.MIDI_NOISE_FLOOR:
        return []
    if note_index < midi.SILENT_NOTES:
        return []
    # Convert note_index in array to actual note-value
    # note = midi.LOWEST_NOTE + note_index - midi.SILENT_NOTES
    note = _note(note_index)
    if note > midi.HIGHEST_NOTE:
        config.debug('decode_note: note index > highest note')
    return g.note_on_off(c, note + transpose, 127, t)


def _note(note_index):
    i = note_index - midi.SILENT_NOTES
    return midi.USED_PITCHES[i][0]
    # note = pitches.USED_DRUMS
    # for i, note_list in enumerate(pitches.DRUMS):
    #     if value in note_list:
    #         return midi.SILENT_NOTES + i
