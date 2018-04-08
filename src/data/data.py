""" Functions that are specific to our dataset
"""
import os, sklearn, skimage, skimage.io, pandas, numpy as np
import pandas, collections
import keras.utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import config

# from utils import utils


# TODO omit channel info?
def midi_to_matrix(midi):
    ls = []
    for msg in midi:
        print('is_meta: %s | bytes():' % msg.is_meta, msg.bytes())
        print(msg)
        if not msg.is_meta:
            ls.append(msg.bytes())
    return np.array(ls)


def make_compatible(arrays):
    # :arrays :: list np.array(n,3)
    smallest = arrays[0].shape[0]
    for a in arrays:
        # TODO check dimension priority
        if a.shape[0] < smallest:
            smallest = a.shape[0]
    return [a[0:smallest] for a in arrays]
