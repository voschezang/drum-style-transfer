from __future__ import division

import numpy as np, collections
from typing import List, Dict
import itertools
import json

import config
import errors
import models
import compression
from utils import utils

### --------------------------------------------------------------------
### NCD-based evaluations
### --------------------------------------------------------------------


def cross(z,
          genre_dict,
          transformations,
          generator,
          grid=[0, 0.5, 1],
          different_genre_a=True,
          amt1=None,
          amt2=None,
          v=0):
    """
    transformations :: {genre a: {genre b: z}}
    genre_dict = {'genre': indices}
    x = images/midi-matrices
    z = latent vector
    generator has method generator.predict(z) -> x

    Make sure that z, genre_dict and transformations are compatible
    all transformations.keys should be in genre_dict.keys
    all genre_dict.values should be in z


    results = {'original genre' : {genre a': {'genre b': grid_result}}}
    grid_result = {'scalar': ncd()}
    """
    sample_size = 1
    results = {}
    if v: print('different_genre_a =', different_genre_a)
    if amt1:
        iter_ = np.array(list(genre_dict.keys()))
        np.random.shuffle(iter_)
        iter_ = iter_[:amt]
    else:
        iter_ = genre_dict.keys()

    for original_genre in iter_:
        if v: print('\noriginal genre: `%s`' % original_genre)
        # TODO non-global ncd-s?
        # for i in range(min(sample_size, len(indices))):
        result = for_every_genre(
            z,
            original_genre,
            genre_dict,
            transformations,
            generator,
            grid,
            different_genre_a,
            amt2,
            v=v)

        if result:
            results[original_genre] = result
    return results


def for_every_genre(z,
                    original_genre,
                    genre_dict,
                    transformations,
                    generator,
                    grid=[0, 1],
                    different_genre_a=True,
                    amt=None,
                    v=0):
    """
    'grid search' of the NCD of original_genre to 'genre b' for 'amt' transformations
    set `different_genre_a` to True to use 'complex' tranformations,
      i.e. to apply A->B to C, with C = original
    set `different_genre_a` to False to use 'simple' tranformations,
      i.e. to mix orignal_genre with genre_b
      i.e. to apply C->B to C, with C = original

    result = {genre a': {'genre b': grid_result}}}
    grid_results = {scalar: ncd()}
    """
    result = {}
    z_original = z[genre_dict[original_genre]]

    if not different_genre_a:
        if original_genre in transformations.keys():
            # iter1 = itertools.repeat(original_genre, amt)
            iter1 = [original_genre]
        else:
            if v: print('no tranformation found for original genre')
            iter1 = []
    elif amt:
        iter1 = list(transformations.keys())
        np.random.shuffle(iter1)
        iter1 = iter1[:amt]
    else:
        iter1 = transformations.keys()

    for genre_a in iter1:
        if v > 1: print(' genre_a=%s' % genre_a)
        if (not different_genre_a) or (genre_a != original_genre):
            # if not (different_genre_a and genre_a == original_genre):
            # if genre_a != original_genre or not different_genre_a:
            result_genre_a = {}
            if amt:
                iter2 = list(transformations[genre_a].keys())
                np.random.shuffle(iter2)
                iter2 = iter2[:amt]
            else:
                iter2 = transformations[genre_a].keys()

            if v: print(' genre_a `%s`, %i' % (genre_a, len(list(iter2))))
            # for genre_b, transformation in transformations[genre_a].items():
            for genre_b in iter2:
                transformation = transformations[genre_a][genre_b]
                if genre_b != original_genre:
                    if v > 1: print(' - genre_b `%s`' % genre_b)
                    indices_b = genre_dict[genre_b]
                    x_b = generator.predict(z[indices_b])
                    result_genre_a[genre_b], _ = grid_search(
                        z_original, x_b, transformation, generator, grid)
                    # TODO compute result of ncd (original, genre a)

            result[genre_a] = result_genre_a
    return result


def grid_search(z,
                x_other,
                transformation,
                generator,
                grid=[0, 1],
                save_transformed=False):
    """
    result :: {scalar: ncd(_, scalar)}
    transformed = [prediction(_) for all scalars]
    """
    ncd = not save_transformed
    result = {}
    transformed = []
    for value in grid:
        z_ = models.apply_transformation(z, transformation, value)
        x_generated = generator.predict(z_)
        if save_transformed:
            transformed.append(x_generated)
        if ncd:
            result[value] = compression.NCD_multiple(x_generated, x_other)
    return result, transformed


### --------------------------------------------------------------------
### Other (tranformations without ncd)
### --------------------------------------------------------------------


def transform(z,
              genre_dict,
              transformations,
              generator,
              grid=[0, 0.01, 0.1, 0.5, 1],
              amt1=None,
              amt2=None,
              v=0):
    """Apply all transformations, sample based
    transformations :: {genre a: {genre b: z}}
    genre_dict = {'genre': indices}
    x = list of images/midi-matrices
    z = list of latent vector
    generator has method generator.predict(z) -> x

    Make sure that z, genre_dict and transformations are compatible
    all transformations.keys should be in genre_dict.keys
    all genre_dict.values should be in z

    result = {genre a': {'genre b': index of x_result}}
    x_result = list of grid_result
    grid_result :: (grid_scalar, x)
    lookup_table :: {i : (genre_a, genre_b)}
    meta = {genre a},{genre b}
    """
    result = collections.defaultdict(dict)
    x_result = []
    lookup_table = {}
    meta = collections.defaultdict(dict)
    i = 0
    if amt1:
        iter1 = list(transformations.items())[:amt1]
    else:
        iter1 = transformations.items()
    for genre_a, d in iter1:
        if amt2:
            iter2 = list(d.items())[:amt2]
        else:
            iter2 = d.items()
        for genre_b, transformation in iter2:
            if v: print('%s \t-> %s' % (genre_a, genre_b))
            indices_a = genre_dict[genre_a][genre_b]
            z_samples = z[indices_a]
            x_b = None
            _ncd_result, x_transformed = grid_search(
                z_samples,
                x_b,
                transformation,
                generator,
                grid,
                save_transformed=True)
            result[genre_a][genre_b] = i
            x_result.append(x_transformed)
            lookup_table[i] = (genre_a, genre_b)
            meta[genre_a][genre_b] = {i: grid}
            i += 1

    return result, x_result, lookup_table, meta
