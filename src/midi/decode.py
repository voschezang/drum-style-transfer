from __future__ import absolute_import
from __future__ import division

import numpy as np
import collections
import mido
from typing import List, Dict

import config
import errors
import midi
from midi import pitches
from utils import utils
# import midi
# from .. import config
# from .. import errors
# from ..utils import utils
from midi import generators as g
from midi import NoteVector, MultiTrack, Track

# from ..midi import midi
# from ..midi.midi import Note, Notes, Track, MultiTrack


def decode_track(c, matrix: MultiTrack) -> mido.MidiTrack:
    # c :: data.Context
    # matrix :: [ vector per instance ]
    # vector :: [ notes ]

    if not isinstance(matrix, MultiTrack):
        config.debug('decode_track - input was not MultiTrack.',
                     'Assuming MultiTrack')
    # if isinstance(matrix, Track):
    #     multi = False
    # elif isinstance(matrix, MultiTrack):
    #     multi = True
    # else:
    #     config.debug('decode_track - input was not Track | MultiTrack.',
    #                  'Assuming MultiTrack')
    #     multi = True

    # decode notes for each instance
    track = mido.MidiTrack()
    # msgs = []
    t = 0
    for i, vector in enumerate(matrix):
        # lookahead_matrix = the part of the matrix that occurred within
        # 'PADDING' cells before 'i'
        lookahead_matrix = matrix[i - midi.PADDING:i]
        # msgs :: mido.Message, with absolute t in seconds
        msgs = notes(c, NoteVector(vector), t, lookahead_matrix)
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


def notes(c, notes: NoteVector, t,
          lookahead_matrix=None) -> List[mido.Message]:
    # :t :: seconds
    # msg.time = absolute, in seconds
    if not isinstance(notes, NoteVector):  # np.generic
        errors.typeError('numpy.ndarray', notes)
    msgs = []
    for note_index, velocity in enumerate(notes):
        if lookahead_matrix is None or lookahead_matrix[:, note_index].max(
        ) < midi.MIDI_NOISE_FLOOR:
            msgs.extend(note(c, note_index, velocity, t))
    return msgs


def note(c, note_index, velocity, t):
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
    return g.note_on_off(c, note_index, 127, t)


def _note(note_index):
    i = note_index - midi.SILENT_NOTES
    return pitches.USED_PITCHES[i][0]
    # note = pitches.USED_DRUMS
    # for i, note_list in enumerate(pitches.DRUMS):
    #     if value in note_list:
    #         return midi.SILENT_NOTES + i
