from __future__ import division

import numpy as np
from typing import List, Dict

import config
import errors
import models
import compression
from utils import utils

### --------------------------------------------------------------------
### NCD-based evaluations
### --------------------------------------------------------------------


def cross(z, genre_dict, transformations, decoder, amt=None, v=0):
    """
    transformations :: {genre a: {genre b: z}}
    genre_dict = {'genre': indices}
    x = images/midi-matrices
    z = latent vector
    decoder has method decoder.predict(z) -> x

    results = {'original genre' : {genre a': {'genre b': grid_result}}}
    grid_result = {'scalar': ncd()}
    """
    sample_size = 1
    grid = [0, 0.5, 1]  # [0, 0.25, 0.5, 0.75, 1, -1, -0.5]
    results = {}
    if amt:
        iter_ = list(genre_dict.keys())[:amt]
    else:
        iter_ = genre_dict.keys()

    for original_genre in iter_:
        if v: print('\noriginal genre: `%s`' % original_genre)
        # TODO non-global ncd-s?
        # for i in range(min(sample_size, len(indices))):
        result = for_every_genre(z, original_genre, genre_dict,
                                 transformations, decoder, grid, v)
        results[original_genre] = result
    return results


def for_every_genre(z,
                    original_genre,
                    genre_dict,
                    transformations,
                    decoder,
                    grid=[0, 1],
                    v=0):
    """
    'grid search' of the NCD of original_genre to 'genre b' for all transformations

    result = {genre a': {'genre b': grid_result}}}
    grid_results = {scalar: ncd()}
    """
    result = {}
    z_original = z[genre_dict[original_genre]]
    for genre_a in transformations.keys():
        #     for genre_a, indices_a in genre_dict.items():
        if genre_a != original_genre:
            if v:
                print(' genre_a `%s`' % genre_a)
            result_genre_a = {}
            for genre_b, transformation in transformations[genre_a].items():
                if genre_b != original_genre:
                    indices_b = genre_dict[genre_b]
                    x_b = decoder.predict(z[indices_b])
                    result_genre_a[genre_b] = grid_search(
                        z_original, x_b, transformation, decoder, grid)
                    # TODO compute result of ncd (original, genre a)
            result[genre_a] = result_genre_a
    return result


def grid_search(z, x_other, transformation, decoder, grid=[0, 1]):
    result = {}
    for value in grid:
        z_ = models.apply_transformation(z, transformation, value)
        x_decoded = decoder.predict(z_)
        result[value] = compression.NCD_multiple(x_decoded, x_other)
    return result
