import collections
import numpy as np
import sklearn.ensemble
"""
For every g in genres,
    For every 'other' genre,
    the average of all transformations from every sample in g to (the average of the encoded samples in other genre)
i.e.
  mean(transformations(a -> mean(b)))

Note that the naive assumption is made that all samples per class are clustered.


usage:

best_dims, importances, transformations, min_transformations = \
    transformation_extraction.between_genres(data, labels)
"""

seed = 123


def between_genres(x, genres=[''], v=0):
    # transformations :: {'genre A': {'genre B': vector}}
    transformations = {}
    min_transformations = {}
    best_dims = []
    importances = []
    # reduced_genres :: ['genre-subgenre']
    reduced_genres = [genre[-2] + '-' + genre[-1] for genre in genres][:100]
    for genre in set(reduced_genres):
        if v > 0:
            print('\nGenre A: %s' % genre)
        best_dims_, importances_, transformations_to, min_transformations_to = \
            transformations_from_genre(genre, reduced_genres, x, v)
        best_dims += best_dims_
        importances += importances_
        transformations[genre] = transformations_to
        min_transformations[genre] = min_transformations_to

    return best_dims, importances, transformations, min_transformations


def transformations_from_genre(original_genre, genres, x, v=0):
    # return [best_dim], [importance], {'genre':vector}, {'genre':vector}
    indices_original = []
    others = collections.defaultdict(list)  # {genre: [index]}
    for i, genre in enumerate(genres):
        if genre == original_genre:
            indices_original.append(i)
        else:
            others[genre].append(i)

    best_dims = []
    importance_list = []
    transformations_to = {}  # {'other genre': vector}
    min_transformations_to = {}  # {'other genre': minimal vector}
    for genre_B, indices_B in others.items():
        i, value, t, min_t = _transformation_ab(indices_original, indices_B, x)
        if v > 0:
            print(' genre B: \t%s (len: %i)' % (genre_B, len(indices_B)))
            print(' - i: %i, importance: %f' % (i, value))
        best_dims.append(i)
        importance_list.append(value)
        transformations_to[genre_B] = t
        min_transformations_to[genre_B] = min_t
    return best_dims, importance_list, transformations_to, min_transformations_to


def _transformation_ab(indices_A, indices_B, x):
    X, y = build_Xy(x, indices_A, indices_B)
    # TODO shuffle?
    i, value = best_feature(X, y)
    t = average_transformation(x[indices_A], x[indices_B])
    min_t = minimal_transformation(t, i)
    return i, value, t, min_t


def best_feature(X, y, n_estimators=250):
    c = sklearn.ensemble.RandomForestClassifier(
        n_estimators, random_state=seed)
    # c = sklearn.ensemble.ExtraTreesClassifier(n_estimators, random_state=seed)
    c.fit(X, y)
    importances = c.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in c.estimators_], axis=0)
    # indices = np.argsort(importances)[::-1]
    i = np.argmax(c.feature_importances_)
    return i, c.feature_importances_[i]


def build_Xy(data, indices_A=[], indices_B=[]):
    indices = indices_A + indices_B
    X = data[indices_A + indices_B]
    y = [0 for _ in indices_A] + [1 for _ in indices_B]
    return X, y


def average_transformation(A, B):
    # return the average of all transformations from a in A to average(B)
    t_per_sample = [np.mean(B, axis=0) - a for a in A]
    t = np.mean(t_per_sample, axis=0)
    return t


def minimal_transformation(vector, i):
    v_ = np.zeros(vector.shape[0])
    v_[i] = vector[i]
    return v_
