"""
Approximations for the NCD

This implementation does not scale well in terms of performance and precision
"""
from __future__ import division

import numpy as np, bz2
import mido
from typing import List, Dict

import config
import errors
import midi
from utils import utils

### --------------------------------------------------------------------
### NCD computation
### --------------------------------------------------------------------


def K(x):
    # approximate the Kolmogorov complexity of x
    if isinstance(x, str):
        return len(bz2.compress(x.encode('utf-8')))
    return K(to_string(x))


def K_conditional(x, y):
    # K(x|y) = K(xy) - K(x)
    # according to the `symmetry of information` theorem by Li and Vitanyi (1997)
    return K(x + y) - K(x)


def NCD(x, y, v=0):
    # Normalized Compression Distance
    if x is y:
        if v:
            print('Warning: x is y')
        return 0.
    return max(K_conditional(x, y), K_conditional(y, x)) / max(K(x), K(y))


def NCD_multiple(xs, ys, pre_concatenation=False, separator='/', v=0):
    if pre_concatenation:
        # concatenate midi matrices
        x = to_string(midi.concatenate(xs))
        y = to_string(midi.concatenate(ys))
    else:
        x, y = '', ''
        for x_ in xs:
            x += separator + to_string(x_)
        for y_ in ys:
            y += separator + to_string(y_)
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
