""" A collection of generic functions that do not have their own modules
"""


def round_(value):
    # return an int, regardless of the input type
    # (even if type(input) is np.float)
    return int(round(value))


def max_f(dt):
    # return the highest frequency that a sampler with sample rate (1/dt) can record
    #: dt = delta time (sampling interval)
    return (1.0 / dt) / 2.0


def min_f(max_t):
    # return the lowest frequency that a sampler with sample rate (1/dt) can record
    return 1.0 / max_t
