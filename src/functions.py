""" Function that extract time-related features from arrays
"""

import numpy as np, collections


def extract_frequencies_matrix(matrix, floor=0, default=0):
    # return a summary of the frequencies per row
    return np.apply_along_axis(
        lambda a: summarize_array(extract_frequencies(a, floor=floor), axis=0),
        1, matrix)


def extract_max_frequencies_matrix(matrix, floor=0, default=0):
    return np.apply_along_axis(
        lambda a: extract_frequencies(a, floor=floor).max(), 1, matrix)


def extract_frequencies(array, floor=0, default=0, increase_recall=False):
    # array = 1 dim array
    # return a list of frequencies
    # return empty array when no signals are detected
    indices = np.where(array > floor)[0]
    if indices.shape == (0, ):
        return indices
    frequencies = []
    count = 0
    for value in array:
        if count == 0:
            if value > floor:
                count = 1
        else:
            if not value > floor:
                count += 1
            else:
                frequencies.append(1.0 / count)
                count = 1  # restart counting, start with 1
    if increase_recall and count == 1:
        # round the last count
        # this decreases the precision, but allows the algorithm to measure
        # lower frequencies (with min. 1 occurence)
        frequencies.append(1.0 / count)
    return np.array(frequencies)


def extract_frequencies_tolerant(array, floor=0) -> np.ndarray:
    # measure frequencies of signals that occur min. once in the input window (array)
    return keep_first_hits(array, floor, increase_recall=True)


def keep_first_hits(array, axis=1) -> np.ndarray:
    # SDT
    # if multiple measurements are found, return all but the last
    # else return all
    if array.shape[1] > 1:
        return array[:-1]
    else:
        return array


def summarize_array(array, axis=1):
    return np.stack([
        array.mean(axis=axis),
        array.min(axis=axis),
        array.max(axis=axis),
    ])


# extract_frequencies(np.array([1, 0, 0, 1, 0, 0]))
# extract_frequencies(np.array([1, 0, 0, 1, 0]))
