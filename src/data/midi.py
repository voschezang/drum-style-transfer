""" Functions that have to do with midi files
Midi is represented either as mido.midi or np.array

The encoding is not lossless: the data is quantized and meta-data is discarded

Encoding & decoding (mido-np conversion)
During encoding, time is automatically converted, dependent on the meta info of the midifile
During decoding, midi-time (PPQ) is set according to the definitions in `context`.
Thus, the midi-time resolution may change during conversion.

Note-off messages are ignored during encoding, for they often are independent of the actual length of a sound. (The length/decay of the sound of the majority of percussion instruments is determined by the instrument itself, and not by the player)

"""

import numpy as np
import mido, rtmidi  #, rtmidi_
from typing import List, Dict

import config, errors
from utils import utils

SILENT_NOTES = 0  # 0: no silent notes | int: silent notes
LOWEST_NOTE = 60
HIGHEST_NOTE = 62
N_NOTES = HIGHEST_NOTE - LOWEST_NOTE + SILENT_NOTES
VELOCITY_RANGE = 127
NOTE_OFF = 'note_off'
NOTE_ON = 'note_on'
MIDI_NOISE_FLOOR = 0.5  # real values below this number will be ignored by midi decoders

# 0.5 to be compatible with binary crossentropy

# def to_matrix(midi):
#     # midi :: mido.midi
#     ls = []
#     for msg in midi:
#         print('is_meta: %s | bytes():' % msg.is_meta, msg.bytes())
#         print(msg)
#         if not msg.is_meta:
#             ls.append(msg.bytes())
#     return _normalize(np.array(ls))


class Notes(np.ndarray):
    # array with floats in range [0,1] for every (midi) note
    # to be used as note-on messages at an instance
    # def __new__(cls, array=np.zeros(N_NOTES)):
    # note: function default args are evaluated once, before runtime
    def __new__(cls, array=None):
        if array is None:
            array = np.zeros(N_NOTES)
        return array.view(cls)


class Track(np.ndarray):
    # array of Notes, with length 'track-length'
    def __new__(cls, length, dt):
        arr = np.stack([Notes() for _ in range(length)])
        # at every timestep, fill notes with index in range 0:SILENT_NOTES with 1
        arr[:, :SILENT_NOTES] = 1.
        return arr.view(cls)

    def __init__(self, length=100, dt=0.01):
        self.dt = dt  # seconds

    def length(self):
        # n instances * dt, in seconds
        return self.shape[0] * self.dt


def solo():
    # extract a single track from a mido.MidiFile
    pass


def encode(c, midi, stretch=False):
    if not isinstance(midi, mido.MidiFile):  # np.generic
        errors.typeError('mido.MidiFile', midi)
    # c :: data.Context
    # TODO # if bpm is given: 'normalize' t

    # matrix :: [ [notes] per instance ]
    matrix = Track(c.n_instances, c.dt)
    t = 0

    # length = midi.length # in seconds
    midi.ticks_per_beat  # e.g. 96 PPQ pulses per quarter note (beat)
    # default tempo: 500000 microseconds per beat (120 bpm)
    #   use set_tempo to change tempo during a song
    # mido.bpm2tempo(bpm)

    # a midifile that consists of multiple tracks is interpreted as multiple independent files
    if midi.type == 2:
        # type = async
        errors.typeError('mido.MidiFile.type 0 | 1', 2)
    elif midi.type == 1:
        # config.debug('WARNING', 'type not == 0')
        print('WARNING', 'type not == 0')
    #     midis = midi.tracks
    # elif midi.type == 0:
    #     midis = [midi]

    # this auto-converts midi msgs.time to seconds
    # alternative: use
    #  for i, track in midi.tracks
    #    msg = track[index]
    for msg in midi:
        t += msg.time  # seconds for type 1,2
        i = utils.round_(t / c.dt)  # instance index (time-step)
        if i < c.n_instances:
            # if i <= c.n_instances:
            # # prevent too high i due to rounding errors
            # if i == c.n_instances: i -= 1
            vector = encode_msg(msg)
            # matrix[i, ] = matrix[i]
            # result = combine_vectors(matrix[i], vector)
            matrix[i, ] = combine_notes(matrix[i], vector)
        else:
            config.debug('to_array: msg.time > max_t; t, n', t, c.n_instances)
            # max t reached: return matrix
            return matrix
    return matrix


def decode_track(c, matrix: Track) -> mido.MidiTrack:
    # c :: data.Context
    # matrix :: [ vector per instance ]
    # vector :: [ notes ]

    # decode notes for each instance
    track = mido.MidiTrack()
    # msgs = []
    t = 0
    for i, vector in enumerate(matrix):
        # msgs :: mido.Message, with absolute t in seconds
        msgs = decode_notes(c, Notes(vector), t)
        for msg in msgs:
            track.append(msg)
        t += c.dt

    # convert absolute time in seconds to relative ticks
    track.sort(key=lambda msg: msg.time)
    track = convert_time_to_relative_value(track, lambda t: second2tick(c, t))

    mid = mido.MidiFile()
    mid.ticks_per_beat = c.ticks_per_beat
    mid.tracks.append(track)
    config.info('len, max_t', mid.length, c.max_t)
    return mid


# def encode_msg(msg: mido.Message) -> Notes:
def encode_msg(msg):
    # midi :: mido midi msg
    # TODO
    # ignore msg.velocity for now
    notes = 0
    notes = Notes()
    # TODO
    # for each instance
    #   for each channel: [note], [velocity]
    if msg.is_meta:
        # config.debug('to_vector: msg is meta')
        return notes
    # ignore note_off TODO
    if msg.type == NOTE_ON:
        normalized_note = max(min(msg.note, HIGHEST_NOTE), LOWEST_NOTE)
        note_index = SILENT_NOTES + normalized_note - LOWEST_NOTE - 1
        notes[note_index] = 1.
    return notes


def decode_notes(c, notes: Notes, t) -> List[mido.Message]:
    # :t :: seconds
    # msg.time = absolute, in seconds
    if not isinstance(notes, Notes):  # np.generic
        errors.typeError('numpy.ndarray', notes)
    msgs = []
    for note_index, value in enumerate(notes):
        if note_index >= SILENT_NOTES:
            note = LOWEST_NOTE + note_index - SILENT_NOTES
            if note > HIGHEST_NOTE:
                config.debug('decode_note: note value > highest note')
            if value > MIDI_NOISE_FLOOR:
                # value *= RANGE
                msg1 = mido.Message(NOTE_ON, note=note, velocity=127, time=t)
                msg2 = mido.Message(
                    NOTE_OFF, note=note, velocity=127, time=t + c.note_length)
                msgs.append(msg1)
                msgs.append(msg2)
    return msgs


def second2tick(c, t):
    return round(mido.second2tick(t, c.ticks_per_beat, c.tempo))


def combine_notes(v1, v2):
    v = Notes((v1 + v2).clip(0, 1))
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
