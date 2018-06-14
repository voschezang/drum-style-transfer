import collections
import numpy as np, os
import sklearn.ensemble
"""
For every g in genres,
    For every 'other' genre,
    the average of all transformations from every sample in g to (the average of the encoded samples in other genre)
i.e.
  mean(transformations(a -> mean(b)))

Note that the naive assumption is made that all samples per class are clustered.


# Datatypes

:transformations :: {genre a: { genre b: transformation}}
:genre :: 'genre/subgenre'

# Usage:

best_dims, importances, transformations, min_transformations = \
    transformation_extraction.between_genres(data, labels)
"""

seed = 123

from utils import io

# def info():
# return "dict/csv per subgenre A :: {'genre B/subgenre B': vector}"

# def save_to_disk(transformations={}, dn='', v=0):
#     with open(dn + 'info.txt', "w") as text_file:
#         print(info(), file=text_file)

#     for genre_A, sub_dict in transformations.items():
#         # sub_dict :: {'target_genre_2/genre_B_2': vector}
#         a1, a2 = genre_A.split('/')
#         if a1 not in os.listdir(dn): os.mkdir(dn + a1)
#         if v:
#             print(genre_A)
#             print(sub_dict.keys())
#         io.save_dict(dn + a1, a2, sub_dict)


def between_genres(x, genre_dict, amt=None, v=0):
    # genre_dict :: {genre: indices}
    # transformations :: {'genre A': {'genre B': vector}}
    transformations = {}
    min_transformations = {}
    best_dims = []
    importances = []
    if amt:
        iter_ = list(genre_dict.items())[:amt]
    else:
        iter_ = genre_dict.items()
    for genre, indices in iter_:
        if v > 0:
            print('\n Genre A: %s' % genre)
        best_dims_, importances_, transformations_to, min_transformations_to = \
            transformations_from_genre(genre, genre_dict, x, amt, v)
        best_dims += best_dims_
        importances += importances_
        transformations[genre] = transformations_to
        min_transformations[genre] = min_transformations_to

    return best_dims, importances, transformations, min_transformations


def transformations_from_genre(original_genre, genre_dict, x, amt=None, v=0):
    # return [best_dim], [importance], {'genre':vector}, {'genre':vector}
    original_indices = genre_dict[original_genre]
    if max(original_indices) >= x.shape[0]:
        print('original_indices >= x.shape', max(original_indices), x.shape[0])
    best_dims = []
    importance_list = []
    transformations_to = {}  # {'other genre': vector}
    min_transformations_to = {}  # {'other genre': minimal vector}
    if amt:
        iter_ = list(genre_dict.items())[:amt]
    else:
        iter_ = genre_dict.items()
    for target_genre, target_indices in iter_:
        if not original_genre == target_genre:
            if max(target_indices) >= x.shape[0]:
                print('target_indices >= x.shape', max(target_indices),
                      x.shape[0])
            i, value, t, min_t = _transformation_ab(original_indices,
                                                    target_indices, x)
            if v > 0:
                print('  genre B: \t%s (len: %i)' % (target_genre,
                                                     len(target_indices)))
                print(' \t i: %i, importance: %f' % (i, value))
            best_dims.append(i)
            importance_list.append(value)
            transformations_to[target_genre] = t
            min_transformations_to[target_genre] = min_t
    return best_dims, importance_list, transformations_to, min_transformations_to


def _transformation_ab(indices_a, indices_b, x):
    X, y = build_Xy(x, indices_a, indices_b)
    # TODO shuffle?
    i, value = best_feature(X, y)
    t = average_transformation(x[indices_a], x[indices_b])
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


def build_Xy(data, indices_a=[], indices_b=[]):
    indices = np.concatenate([indices_a, indices_b], axis=-1)
    X = data[indices]
    y = [0 for _ in indices_a] + [1 for _ in indices_b]
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
