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


def cross(z, genre_dict, transformations, generator, amt1=None, amt2=None,
          v=0):
    """
    transformations :: {genre a: {genre b: z}}
    genre_dict = {'genre': indices}
    x = images/midi-matrices
    z = latent vector
    generator has method generator.predict(z) -> x

    results = {'original genre' : {genre a': {'genre b': grid_result}}}
    grid_result = {'scalar': ncd()}
    """
    sample_size = 1
    grid = [0, 0.5, 1]  # [0, 0.25, 0.5, 0.75, 1, -1, -0.5]
    results = {}
    if amt1:
        iter_ = np.array(list(genre_dict.keys()))
        i = 0
        while i < amt:
            # iter_ = list(genre_dict.keys())[:amt1]
            # np.random.shuffle(iter_)
            # iter_ = iter_[:amt1]
            original_genre = np.random.choice(iter_)
            if v: print('\noriginal genre: `%s`' % original_genre)
            result = for_every_genre(z, original_genre, genre_dict,
                                     transformations, generator, grid, amt2, v)
            results[original_genre] = result
            i += 1

    else:
        iter_ = genre_dict.keys()

        for original_genre in iter_:
            if v: print('\noriginal genre: `%s`' % original_genre)
            # TODO non-global ncd-s?
            # for i in range(min(sample_size, len(indices))):
            result = for_every_genre(z, original_genre, genre_dict,
                                     transformations, generator, grid, amt2, v)
            results[original_genre] = result
    return results


def for_every_genre(z,
                    original_genre,
                    genre_dict,
                    transformations,
                    generator,
                    grid=[0, 1],
                    amt=None,
                    v=0):
    """
    'grid search' of the NCD of original_genre to 'genre b' for all transformations

    result = {genre a': {'genre b': grid_result}}}
    grid_results = {scalar: ncd()}
    """
    result = {}
    z_original = z[genre_dict[original_genre]]
    if amt:
        # iter1 = list(genre_dict.keys())[:amt]
        iter1 = list(transformations.keys())
        np.random.shuffle(iter1)
        iter1 = iter1[:amt]
    else:
        iter1 = transformations.keys()

    for genre_a in iter1:
        if genre_a != original_genre:
            if v:
                print(' genre_a `%s`' % genre_a)
            result_genre_a = {}
            if amt:
                iter2 = list(transformations[genre_a].keys())
                np.random.shuffle(iter2)
                iter2 = iter2[:amt]
            else:
                iter2 = transformations[genre_a].keys()

            # for genre_b, transformation in transformations[genre_a].items():
            for genre_b in iter2:
                transformation = transformations[genre_a][genre_b]
                if genre_b != original_genre:
                    if v > 1:
                        print(' - genre_b `%s`' % genre_b)
                    indices_b = genre_dict[genre_b]
                    x_b = generator.predict(z[indices_b])
                    result_genre_a[genre_b] = grid_search(
                        z_original, x_b, transformation, generator, grid)
                    # TODO compute result of ncd (original, genre a)
            result[genre_a] = result_genre_a
    return result


def grid_search(z, x_other, transformation, generator, grid=[0, 1]):
    result = {}
    for value in grid:
        z_ = models.apply_transformation(z, transformation, value)
        x_generated = generator.predict(z_)
        result[value] = compression.NCD_multiple(x_generated, x_other)
    return result
