""" Functions that have to do with midi files
Midi is represented either as mido.midi or np.array

The encoding is not lossless: the data is quantized and meta-data is discarded
"""

import numpy as np
import mido, rtmidi  #, rtmidi_

import config

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


def encode(c, midi, stretch=False):
    # c :: data.Context
    # TODO
    # if bpm is given: 'normalize' t

    # matrix :: [ vector per instance ]
    # vector :: [ notes ]
    matrix = np.zeros([c.n_instances, N_NOTES])
    t = 0
    # this auto-converts midi msgs.time to seconds
    # alternative: use
    #  for i, track in midi.tracks
    #    msg = track[index]
    for msg in midi:
        # print(msg.time)
        t += msg.time
        i = int(round(t / c.dt))  # instance index (time-step)
        if i <= c.n_instances:
            vector = encode_vector(msg)
            matrix[i, ] = combine_vectors(matrix[i], vector)
        else:
            config.debug('to_array: msg.time > max_t')
    return matrix


def decode(c, matrix):
    # c :: data.Context
    # matrix :: [ vector per instance ]
    # vector :: [ notes ]
    track = mido.MidiTrack()
    for i, vector in enumerate(matrix):
        t = i * c.dt
        msgs = decode_vector(c, vector, t)
        for msg in msgs:
            track.append(msg)
    mid = mido.MidiFile()
    mid.tracks.append(track)
    return mid


def encode_vector(msg):
    # midi :: mido midi msg
    # TODO
    # ignore msg.velocity for now
    notes = np.zeros(N_NOTES)
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


def decode_vector(c, vector, t=0):
    msgs = []
    for note, value in enumerate(vector):
        if value > 0:
            # value *= RANGE
            msg1 = mido.Message('note_on', note=note, velocity=127, time=t)
            msg2 = mido.Message(
                'note_off', note=note, velocity=127, time=t + c.note_length)
            msgs.append(msg1)
            msgs.append(msg2)
    return msgs


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
