""" Functions to generate midi sequences according to formulas
"""

import numpy as np
import mido

import config
import midi
from midi import encode
from utils import utils

LOWEST_F = 0.01  # 1/f = T: 1/0.01 = 100 s


def example(c):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    t = 0
    for i in range(3):
        t += i * 100
        for note in [60, 62, 63]:
            track.append(
                mido.Message('note_on', note=note, velocity=127, time=t))
            track.append(
                mido.Message(
                    'note_off', note=note, velocity=127, time=t + c.dt))

    track.sort(key=lambda msg: msg.time)
    track = midi.convert_time_to_relative_value(
        track, lambda t: midi.second2tick(c, t))
    mid.tracks.append(track)
    return midi.encode.midiFiles(c, [mid])


def gen_data(c, n=100, fs=None, max_f=None, min_f=None) -> np.ndarray:
    # generate midifiles with a straight note pattern (e.g. 3Hz)
    f_margin = 0.10  # 10%
    if max_f is None:
        max_f = utils.max_f(c.dt) * (1 - f_margin)
    if min_f is None:
        min_f = utils.min_f(c.max_t) * (1 + f_margin)
    if min_f > max_f:
        config.debug('min_f > max_f', min_f, max_f)
    if fs is None:
        fs = np.random.random(n) * (max_f - min_f) + min_f
    midis = [render_midi(c, f, phase=np.random.random()) for f in fs]
    return midi.encode.midiFiles(c, midis)


def gen_data_complex(c,
                     n=100,
                     max_f=None,
                     min_f=None,
                     n_polyrythms=2,
                     n_channels=2,
                     d_phase=True,
                     return_params=False,
                     dim4=False,
                     multiTrack=True):
    """
    :n = n independent samples
    :n_polyrythms = n 'sinewaves' per channel
    :n_channels = n different note indices (e.g. 60,62,64)

    set d_phase to False to let every pattern start at t=0
    """
    f_margin = 0.10  # 10%
    if max_f is None:
        max_f = utils.max_f(c.dt) * (1 - f_margin)
    if min_f is None:
        min_f = utils.min_f(c.max_t) * (1 + f_margin)
    if min_f > max_f:
        config.debug('min_f > max_f', min_f, max_f)

    n_channels = min(n_channels, midi.N_NOTES)
    # params :: (samples, channels, frequencies)
    params = np.random.random([n, n_channels, n_polyrythms]) * (
        max_f - min_f) + min_f
    params[np.where(params < LOWEST_F)] = 0.
    midis = [render_midi_poly(c, ffs, d_phase=d_phase) for ffs in params]
    matrices = midi.encode.midiFiles(c, midis, multiTrack, dim4=dim4)

    if return_params:

        return matrices, params
    return matrices


def render_midi(c, f=1, max_t=10, phase=0, polyphonic=False):
    # generate a midifile with a straight note pattern (e.g. 3Hz)
    # set polyphonic to true to duplicate the pattern to multiple notes
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track = gen_square_wave(c, track, f, max_t, phase, polyphonic)
    track.sort(key=lambda msg: msg.time)
    track = midi.convert_time_to_relative_value(
        track, lambda t: midi.second2tick(c, t))
    mid.tracks.append(track)
    return mid


def render_midi_poly(c, ffs=[[1]], max_t=10, d_phase=True):
    # ffs :: (notes, frequency values)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    note = midi.LOWEST_NOTE
    phase = 0
    for fs in ffs:
        for f in fs:
            if d_phase:
                phase = np.random.random()
            track = gen_square_wave(
                c,
                track,
                f,
                max_t,
                phase_offset=phase,
                polyphonic=False,
                note=note)
        note += 1

    track.sort(key=lambda msg: msg.time)
    track = midi.convert_time_to_relative_value(
        track, lambda t: midi.second2tick(c, t))
    mid.tracks.append(track)
    return mid


def gen_square_wave(c,
                    track=[],
                    f=1,
                    max_t=10,
                    phase_offset=0,
                    polyphonic=True,
                    note=None) -> []:
    # Generate a sequence of notes, spaced evenly over time
    # The pulse-width is defined by c.note_length
    # :track :: list | mido.MidiTrack
    notes = gen_note_values(polyphonic, note)
    if f < LOWEST_F:
        # Return a single note
        for note in notes:
            track.extend(midi.gen_note_on_off(c, note, 127, t=0))
        return track

    dt = 1. / f
    start_t = dt * phase_offset
    t = start_t  # absolute t in seconds
    while t < max_t:
        for note in notes:
            track.extend(midi.gen_note_on_off(c, note, 127, t))

        t += dt

    return track


def gen_note_values(polyphonic=False, note=None):
    if polyphonic:
        # Generate all possible notes
        notes = np.arange(midi.LOWEST_NOTE, midi.HIGHEST_NOTE)
    else:
        if note:
            notes = np.array([note])
        else:
            notes = np.array([midi.LOWEST_NOTE])

    return notes + midi.SILENT_NOTES


def render(sin=np.sin, f=1, n_samples=10, dt=0.01, phase=0):
    """
    :sin = any cycling formula
    :f = frequency
    :n_samples = amount of sampled instances
    :dt = sample size
    :phase = int in range [0, 2pi]

    max_t = n_samples * dt
    """
    samples = np.arange(n_samples) * dt
    return ac_to_dc(sin(2 * np.pi * f * samples + 2 * np.pi * phase))


def normalize(array):
    return ac_to_dc(array)


def ac_to_dc(array):
    # alternating current to direct current
    # [-1,1] -> [0,1]
    return (array + 1) * 0.5
