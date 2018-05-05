""" Functions that have to do with midi files
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

import numpy as np, collections
import mido, rtmidi  #, rtmidi_
from typing import List, Dict

import config, errors
from utils import utils

SILENT_NOTES = 0  # 0: no silent notes | int: silent notes
LOWEST_NOTE = 50
HIGHEST_NOTE = 80
N_NOTES = HIGHEST_NOTE - LOWEST_NOTE + SILENT_NOTES
VELOCITY_RANGE = 127
NOTE_OFF = 'note_off'
NOTE_ON = 'note_on'
MIDI_NOISE_FLOOR = 0.5  # real values below this number will be ignored by midi decoders
PADDING = 4  # n array cells after an instance of a note-on msg (should be > 0)
VELOCITY_DECAY = 0.6  # velocity decay for every padded-cell

DTYPE = 'float32'

# 0.5 to be compatible with binary crossentropy


class Note(np.ndarray):
    # array :: [0] | [1]
    # array with a single float in range [0,1]
    # to be used as a note-on message at an instance
    # note: function default args are evaluated once, before runtime
    def __new__(cls, array=None):
        if array is None:
            array = np.zeros(1, dtype=DTYPE)
        return array.astype(DTYPE).view(cls)


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


# class Track(np.ndarray):
#     # array of Notes, with length 'track-length'
#     def __new__(cls, length, dt):
#         arr = np.stack([Note() for _ in range(length)])
#         return arr.view(cls)

#     def __init__(self, length=100, dt=0.01):
#         self.dt = dt  # seconds

#     def length_in_seconds(self):
#         # n instances * dt, in seconds
#         return self.shape[0] * self.dt


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


def solo():
    # TODO
    # extract a single track from a mido.MidiFile
    pass


# def notes_to_notelist(notes_list):
#     # notes :: array (samples, timesteps, notes)
#     # return array (samples * notes, timesteps, note-value)
#     notelist_list = [multiTrack_to_list_of_Track(m) for m in notes_list]
#     return np.concatenate(notelist_list)


def encode_midiFiles(c,
                     midis,
                     multiTrack=True,
                     reduce_dims=True,
                     force_velocity=False):
    ls = [
        encode_midiFile(
            c, m, multiTrack=multiTrack, force_velocity=force_velocity)
        for m in midis
    ]
    if multiTrack:
        tracks = np.stack(ls)
    else:
        tracks = np.concatenate(ls)

    if reduce_dims:
        indices = []
        print(tracks.shape)
        for note_i in np.arange(tracks.shape[-1]):
            if tracks[:, :, note_i].max() > MIDI_NOISE_FLOOR:
                indices.append(note_i)
        tracks = tracks[:, :, indices]
        config.info('reduced dims:', tracks.shape)

    return tracks


def encode_midiFile(c,
                    midi,
                    stretch=False,
                    squash=False,
                    multiTrack=True,
                    force_velocity=None):
    # TODO stretch, squash
    if not isinstance(midi, mido.MidiFile):
        errors.typeError('mido.MidiFile', midi)
    # c :: data.Context
    # TODO # if bpm is given: 'normalize' t ?

    # matrix :: [ [notes] per instance ]
    # all midinotes will be grouped into 1 MultiTrack per midichannel
    matrix = MultiTrack(c.n_instances, c.dt)
    t = 0

    # length = midi.length # in seconds
    midi.ticks_per_beat  # e.g. 96 PPQ pulses per quarter note (beat)
    # default tempo: 500000 microseconds per beat (120 bpm)
    #   use set_tempo to change tempo during a song
    # mido.bpm2tempo(bpm)

    # TODO? a midifile that consists of multiple tracks is interpreted
    # as multiple independent files
    if midi.type == 2:
        # type = async
        errors.typeError('mido.MidiFile.type 0 | 1', 2)
    # elif midi.type == 1:
    # TODO are multiple midichannels concatenated?
    # config.debug('WARNING', 'type not == 0')
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
        if not i < c.n_instances:
            config.debug('to_array: msg.time > max_t; t, n', t, c.n_instances)
            # max t reached: return matrix
            if multiTrack:
                return matrix
            return multiTrack_to_list_of_Track(matrix)

        matrix = encode_msg_in_matrix(c, msg, i, matrix, force_velocity)

    if multiTrack:
        return matrix
    return multiTrack_to_list_of_Track(matrix)


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
        lookahead_matrix = matrix[i - PADDING:i]
        # msgs :: mido.Message, with absolute t in seconds
        msgs = decode_notes(c, Notes(vector), t, lookahead_matrix)
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


def encode_msg_in_matrix(c, msg: mido.Message, i_, matrix, velocity=None):
    # :velocity = None | float in range(0,1)
    if msg.is_meta:
        # config.info('to_vector: msg is meta')
        return matrix

    if velocity is None:
        velocity = min(msg.velocity, VELOCITY_RANGE) / float(VELOCITY_RANGE)

    for i in range(i_, i_ + PADDING):
        if i < c.n_instances:
            vector = encode_single_msg(msg, velocity)
            matrix[i, ] = combine_notes(matrix[i], vector)
            velocity *= VELOCITY_DECAY
    return matrix


def encode_single_msg(msg: mido.Message, velocity=None) -> Notes:
    # encoder mido.Message to vector
    # midi :: mido midi msg
    # TODO
    # ignore msg.velocity for now
    notes = Notes()
    default_note = 1.
    # TODO
    # for each instance
    #   for each channel: [note], [velocity]
    if msg.is_meta:
        # config.info('to_vector: msg is meta')
        return notes
    # ignore note_off TODO
    if msg.type == NOTE_ON:
        if velocity is None:
            velocity = default_note
        highest_note_i = HIGHEST_NOTE - 1
        normalized_note = max(min(msg.note, highest_note_i), LOWEST_NOTE)
        note_index = SILENT_NOTES + normalized_note - LOWEST_NOTE
        notes[note_index] = velocity
    return notes


def decode_notes(c, notes: Notes, t,
                 lookahead_matrix=None) -> List[mido.Message]:
    # :t :: seconds
    # msg.time = absolute, in seconds
    if not isinstance(notes, Notes):  # np.generic
        errors.typeError('numpy.ndarray', notes)
    msgs = []
    for note_index, value in enumerate(notes):
        if lookahead_matrix is None or lookahead_matrix[:, note_index].max(
        ) < MIDI_NOISE_FLOOR:
            msgs.extend(decode_note(c, note_index, value, t))
    return msgs


def decode_note(c, note_index, value, t):
    # return ::  [] | a list of midi messages (note on, note off)
    if value < MIDI_NOISE_FLOOR:
        return []
    if note_index < SILENT_NOTES:
        return []
    note = LOWEST_NOTE + note_index - SILENT_NOTES
    if note > HIGHEST_NOTE:
        config.debug('decode_note: note value > highest note')
    # value *= RANGE
    msg1 = mido.Message(NOTE_ON, note=note, velocity=127, time=t)
    msg2 = mido.Message(
        NOTE_OFF, note=note, velocity=127, time=t + c.note_length)
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
