"""
Approximations for the NCD

This implementation does not scale well in terms of performance and precision

TODO use post-concatenation

"""
from __future__ import division

import numpy as np, bz2
import mido
from typing import List, Dict

import config
import errors
import midi
from utils import utils, io


def K(x: np.array):
    # approximate the Kolmogorov complexity of x
    return len(bz2.compress(to_string(x).encode('utf-8')))


def K_conditional(x, y):
    # K(x|y) = K(xy) - K(x)
    # according to the `symmetry of information` theorem by Li and Vitanyi (1997)
    xy = midi.concatenate([x, y])
    return K(xy) - K(x)


def NCD(x, y, v=0):
    # Normalized Compression Distance
    if x is y:
        if v:
            print('Warning: x is y')
        return 0.
    return max(K_conditional(x, y), K_conditional(y, x)) / max(K(x), K(y))


def NCD_multiple(xs, ys, v=0):
    x = midi.concatenate(xs)
    y = midi.concatenate(ys)
    return NCD(x, y, v)


def to_string(x, suppress_small=False, formatter=True):
    """if formatter:
          . '[[0.8189,0.0000,0.0000,0.7795,0.0000,0'
        else:
          '.    ,0.1528,0.    ,0.    ,0.    ],'
        """
    if len(x.shape) == 3:
        x = x[:, :, 0]
    if formatter:
        return np.array2string(
            x,
            separator=',',
            formatter={'float_kind': lambda x: "%.4f" % x},
            suppress_small=suppress_small,
            threshold=np.nan)

    return np.array2string(
        x,
        precision=4,
        separator=',',
        suppress_small=suppress_small,
        threshold=np.nan)
