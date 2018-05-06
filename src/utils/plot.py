""" Midi-plots
Black indicates a note-on msg
Grey indicates a probable note-on msg (intensity correlates with p())
White indicates a rest
"""

import matplotlib.pyplot as plt

from data import midi


def single(m: midi.MultiTrack):
    # m :: MultiTrack | Track
    print('m', m.shape)
    m = m.transpose()
    # fig, ax = plt.subplots()
    plt.imshow(m, interpolation='nearest', cmap='gray_r')
    # fig.canvas.set_window_title(name + '...')
    # fig.set_title(name)
    # fig.set_xlabel('Time [iterations]')
    # fig.set_ylabel('Score')
    plt.show()


def multi(m):
    return single(m)


def line(matrix):
    plt.plot(matrix[:30, 0])
    plt.show()
