""" Functions that have to do with midi files
"""

import numpy as np
import mido, rtmidi  #, rtmidi_


def to_matrix(midi):
    ls = []
    for msg in midi:
        print('is_meta: %s | bytes():' % msg.is_meta, msg.bytes())
        print(msg)
        if not msg.is_meta:
            ls.append(msg.bytes())
    return np.array(ls)
