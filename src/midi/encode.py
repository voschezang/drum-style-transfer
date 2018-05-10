import numpy as np, collections
import mido
from typing import List, Dict

import config, errors
from utils import utils


def midiFiles(c,
              midis,
              multiTrack=True,
              reduce_dims=True,
              velocity=None,
              dim4=False):
    # reduce dims filters out unused dimensions
    # dim4 adds a dimension, so that the output will fit a keras ImageDataGenerator

    # TODO split long files in samples

    track_list = [
        encode_midiFile(c, m, multiTrack=multiTrack, velocity=velocity)
        for m in midis
    ]
    if multiTrack:
        tracks = np.stack(track_list)
    else:
        tracks = np.concatenate(track_list)

    if reduce_dims:
        indices = []
        for note_i in np.arange(tracks.shape[-1]):
            if tracks[:, :, note_i].max() > MIDI_NOISE_FLOOR:
                indices.append(note_i)
        tracks = tracks[:, :, indices]
        config.info('reduced dims:', tracks.shape)

    if dim4:
        return tracks.reshape(list(tracks.shape) + [1])
    return tracks


def midiFile(c, midi, multiTrack=True, velocity=None, reduce_dims=True):
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
            # config.info('to_array: msg.time > max_t; t, n', t, c.n_instances)
            # max t reached: return matrix
            if multiTrack:
                return matrix
            return multiTrack_to_list_of_Track(matrix)

        matrix = encode_msg_in_matrix(c, msg, i, matrix, velocity)

    # if reduce_dims:
    #     matrix = reduce_dims(matrix)
    #     indices = []
    #     for note_i in np.arange(matrix.shape[-1]):
    #         if tracks[:, :, note_i].max() > MIDI_NOISE_FLOOR:
    #             indices.append(note_i)
    #     tracks = tracks[:, :, indices]
    #     config.info('reduced dims:', tracks.shape)

    if multiTrack:
        return matrix
    return multiTrack_to_list_of_Track(matrix)


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
