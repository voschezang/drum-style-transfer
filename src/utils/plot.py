""" Midi-plots
Black indicates a note-on msg
Grey indicates a probable note-on msg (intensity correlates with p())
White indicates a rest
"""

import matplotlib.pyplot as plt


def single(v):
    # v = v.reshape((v.shape[1], v.shape[0]))
    v = v.transpose()
    # fig, ax = plt.subplots()
    plt.imshow(v, interpolation='nearest', cmap='gray_r')
    # fig.canvas.set_window_title(name + '...')
    # fig.set_title(name)
    # fig.set_xlabel('Time [iterations]')
    # fig.set_ylabel('Score')
    plt.show()


def multi(m):
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


def line(matrix):
    plt.plot(matrix[:30, 0])
    plt.show()
