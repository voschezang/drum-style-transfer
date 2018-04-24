""" Functions to generate midi sequences according to formulas
"""

import numpy as np
import mido, rtmidi  #, rtmidi_

import config
from data import midi
from utils import utils


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
    midis = [
        midi.encode(c, render_midi(c, f, phase=np.random.random())) for f in fs
    ]
    return np.stack(midis)


def gen_data_complex(c, n=100, fs=None, max_f=None, min_f=None) -> np.ndarray:
    f_margin = 0.10  # 10%
    if max_f is None:
        max_f = utils.max_f(c.dt) * (1 - f_margin)
    if min_f is None:
        min_f = utils.min_f(c.max_t) * (1 + f_margin)
    if min_f > max_f:
        config.debug('min_f > max_f', min_f, max_f)
    if fs is None:
        # n = n_samples
        n_polyrythms = 2
        n_channels = 2
        ffs = np.random.random([n, n_polyrythms, n_channels]) * (
            max_f - min_f) + min_f
    midis = [midi.encode(c, render_midi_poly(c, fs)) for fs in ffs]
    return np.stack(midis)


def render_midi(c, f=1, max_t=10, phase=0, polyphonic=True):
    # generate a midifile with a straight note pattern (e.g. 3Hz)
    # set polyphonic to true to duplicate the pattern to multiple notes
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track = add_sin_to_midi_track(c, track, f, max_t, phase, polyphonic)
    track.sort(key=lambda msg: msg.time)
    track = midi.convert_time_to_relative_value(
        track, lambda t: midi.second2tick(c, t))
    mid.tracks.append(track)
    return mid


def render_midi_poly(c, ffs=[[1]], max_t=10):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    for fs in ffs:
        for f in fs:
            phase = np.random.random()
            note = np.random.choice(range(midi.LOWEST_NOTE, midi.HIGHEST_NOTE))
            track = add_sin_to_midi_track(
                c, track, f, max_t, phase, polyphonic=False, note=note)
    track.sort(key=lambda msg: msg.time)
    track = midi.convert_time_to_relative_value(
        track, lambda t: midi.second2tick(c, t))
    mid.tracks.append(track)
    return mid


def add_sin_to_midi_track(c,
                          track,
                          f=1,
                          max_t=10,
                          phase=0,
                          polyphonic=True,
                          note=None):
    dt = 1. / f
    start_t = dt * phase
    t = start_t  # absolute t in seconds

    while t < max_t:
        if note:
            notes = [note]
        elif polyphonic:
            notes = range(midi.LOWEST_NOTE, midi.HIGHEST_NOTE)
        else:
            notes = [midi.LOWEST_NOTE]

        for note in notes:
            note += midi.SILENT_NOTES
            track.append(
                mido.Message('note_on', note=note, velocity=127, time=t))
            track.append(
                mido.Message(
                    'note_off', note=note, velocity=127, time=t + c.dt))
        t += dt

    return track


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
