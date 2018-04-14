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

N_NOTES = 127
RANGE = 127
NOTE_OFF = 'note_off'
NOTE_ON = 'note_on'

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
    # array with value in range [0,1] for every (midi) note
    # to be used as note-on messages at an instance
    def __new__(cls):
        arr = np.zeros(N_NOTES)
        return arr.view(cls)


class Track(np.ndarray):
    # array of Notes, with length 'track-length'
    def __new__(cls, length, dt):
        # values: list of length N_NOTES with values in range
        arr = np.stack([Notes() for _ in range(length)])
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
        config.debug('WARNING', 'type not == 0')
    #     midis = midi.tracks
    # elif midi.type == 0:
    #     midis = [midi]

    # this auto-converts midi msgs.time to seconds
    # alternative: use
    #  for i, track in midi.tracks
    #    msg = track[index]
    for msg in midi:
        t += msg.time  # seconds for type 1,2
        i = round(t / c.dt)  # instance index (time-step)
        if i < c.n_instances:
            # if i <= c.n_instances:
            # # prevent too high i due to rounding errors
            # if i == c.n_instances: i -= 1
            vector = encode_vector(msg)
            # matrix[i, ] = matrix[i]
            # result = combine_vectors(matrix[i], vector)
            matrix[i, ] = combine_vectors(matrix[i], vector)
        else:
            config.debug('to_array: msg.time > max_t', t, c.n_instances)
            # max t reached: return matrix
            return matrix
    return matrix


# def encode_track(c, miditrack, ticks_per_beat, tempo):
#     matrix = np.zeros([c.n_instances, N_NOTES])
#     t = 0
#     ticks_per_beat =
#     # this auto-converts midi msgs.time to seconds
#     # alternative: use
#     #  for i, track in midi.tracks
#     #    msg = track[index]
#     for msg in miditrack:
#         t += msg.time
#         i = round(t / c.dt)  # instance index (time-step)
#         if i <= c.n_instances:
#             # prevent too high i due to rounding errors
#             if i == c.n_instances: i -= 1
#             vector = encode_vector(msg)
#             matrix[i, ] = matrix[i]
#             result = combine_vectors(matrix[i], vector)
#             matrix[i, ] = combine_vectors(matrix[i], vector)
#         else:
#             config.debug('to_array: msg.time > max_t', t, c.n_instances)
#     return matrix


def decode(c, matrix):
    # c :: data.Context
    # matrix :: [ vector per instance ]
    # vector :: [ notes ]
    mid = mido.MidiFile()
    mid.ticks_per_beat = c.ticks_per_beat
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for i, vector in enumerate(matrix):
        t = i * c.dt
        msgs = decode_vector(c, vector, t)
        for msg in msgs:
            track.append(msg)
    return mid


def encode_vector(msg):
    # midi :: mido midi msg
    # TODO
    # ignore msg.velocity for now
    notes = Notes()
    # TODO
    # for each instance
    #   for each channel: [note], [velocity]
    if msg.is_meta:
        # config.debug('to_vector: msg is meta')
        return notes
    # TODO
    # ignore note_off for now
    if msg.type == NOTE_ON:
        notes[msg.note] = 1.
    return notes


def decode_vector(c, vector: np.array, t: float = 0) -> List[mido.Message]:
    # :vector :: instance
    # :t :: seconds
    if not isinstance(vector, np.ndarray):  # np.generic
        errors.typeError('numpy.ndarray', vector)
    msgs = []
    for note, value in enumerate(vector):
        if value > 0:
            ticks = second2tick(c, t)
            ticks2 = second2tick(c, t + c.note_length)
            # value *= RANGE
            msg1 = mido.Message('note_on', note=note, velocity=127, time=ticks)
            msg2 = mido.Message(
                'note_off', note=note, velocity=127, time=ticks2)
            msgs.append(msg1)
            msgs.append(msg2)
    return msgs


def second2tick(c, t):
    return round(mido.second2tick(t, c.ticks_per_beat, c.tempo))


def combine_vectors(v1, v2):
    return (v1 + v2).clip(0, 1)


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
