""" Functions to generate midi sequences according to formulas
"""

import numpy as np
import mido, rtmidi  #, rtmidi_

import config
from data import midi
from utils import utils


def gen_data(c, n=100, fs=None) -> np.ndarray:
    f_margin = 0.10  # 10%
    min_f = utils.min_f(c.max_t) * (1 + f_margin)
    max_f = utils.max_f(c.dt) * (1 - f_margin)
    if min_f < max_f:
        config.debug('min_f < max_f')
    if fs is None:
        fs = np.random.random(n) * (max_f - min_f) + min_f
    midis = [
        midi.encode(c, render_midi(c, f, phase=np.random.random())) for f in fs
    ]
    return np.stack(midis)


def render_midi(c, f=1, max_t=10, phase=0):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    dt = 1. / f
    start_t = dt * phase
    t = start_t  # absolute t in seconds

    while t < max_t:
        track.append(mido.Message('note_on', note=60, velocity=127, time=t))
        track.append(
            mido.Message('note_off', note=60, velocity=127, time=t + c.dt))
        t += dt

    track.sort(key=lambda msg: msg.time)
    track = midi.convert_time_to_relative_value(
        track, lambda t: midi.second2tick(c, t))
    return mid


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
