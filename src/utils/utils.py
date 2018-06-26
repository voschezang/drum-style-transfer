""" A collection of generic functions that do not have their own modules
"""
import numpy as np, collections
import operator as op
import importlib
import scipy, scipy.stats


def reload(package, *args):
    for p in args:
        importlib.reload(p)
    importlib.reload(package)


def get(d: dict, recursion_depth=0, i=-1):
    # use recursion_depth = 1 for a double-dict
    # i.e. {y: {x: ..}}
    keys = []
    if recursion_depth > 0:
        _k, new_keys, d = get(d, recursion_depth - 1, i=i)
        keys.extend(new_keys)

    # pop an item without removal
    k = list(d.keys())[i]
    keys.append(k)
    return k, keys, d[k]


def clean_dict(d: dict, r=0, trash=[[], {}], verbose=0):
    """ rm all trash in a dict
    no deepcopy, thus potential side-effects

    trash :: list of values to remove
    return :: copy of d
    """
    result = {}
    for k, v in d.items():
        if not d[k] in [[], {}]:
            if r > 0:
                v = clean_dict(v, r - 1)
            result[k] = v
        else:
            if verbose: print('found empty item: ', k, d[k])
    return result


# Math


def round_(value):
    # return an int, regardless of the input type
    # (even if type(input) is np.float)
    return int(round(value))


def max_f(dt):
    # return the highest frequency that a sampler with sample rate (1/dt) can record
    #: dt = delta time (sampling interval)
    return (1.0 / dt) / 2.0


def min_f(max_t):
    # return the lowest frequency that a sampler with sample rate (1/dt) can record
    return 1.0 / max_t


def least_squares(y, x=None, line=True, v=0):
    if x is None:
        x = np.linspace(0, 1, len(y))
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    if v:
        print('slope, intercept, r_value, p_value, std_err:', slope, intercept,
              r_value, p_value, std_err)
    if line:
        line = [intercept + slope * a for a in x]
    return line, (slope, intercept, r_value, p_value, std_err)


# Logic


def composition(ls=[], result=lambda x: x, verbose=False):
    # ls = list of functions (pipeline)
    # return :: function composition of ls
    # e.g.
    # this should succeed without errors
    # compose([(lambda x: x) for _ in range(1000000)])(1)
    for f in ls:
        if verbose: print(f)
        result = f(result)
    return result


### ------------------------------------------------------
### Statistics
### ------------------------------------------------------


def summary_multi(data={}, mode=dict):
    """
    data :: {'parameter': [ value ]}
    mode :: dict | list
    return :: {'statistic': {param: score} }
    """
    if mode is dict:
        result = collections.defaultdict(dict)
        iter_ = data.items()
    elif mode is list:
        result = collections.defaultdict(list)
        iter_ = sorted(data.items())
    else:
        raise ('unkown arg `mode`', mode)

    for param, values in iter_:
        summary_result = summary(values)
        for statistic, score in summary_result.items():
            if mode is dict:
                result[statistic][param] = score
            elif mode is list:
                result[statistic].append(score)

    return result


def summary(v=[]):
    return {
        'mean': np.mean(v),
        'median': np.median(v),
        'min': min(v),
        'max': max(v)
    }


def ttest(alpha=0.05, a=[], b=[]):
    s, p = stats.ttest_ind(a, b)
    if p < alpha:
        result = True
        if config.result_: print('RESULT - significant difference found')
    else:
        result = False
        if config.result_: print('RESULT - NO significant difference found')
    return result, s, p


def dict_to_table(data={}, durations={}, print_results=False, txt=''):
    # data :: { 'name': [score] }
    # durations :: { 'name': [time] }
    result_dict = {0: ['best score', 'duration']}
    table = texttable.Texttable()
    rows = [['Algorithm', 'Score', 'Duration']]
    for alg, scores in data.items():
        # select best score
        i, score = max(enumerate(scores), key=lambda x: x[1])
        # add row: name, score, durations for that score
        print(alg, i, score)
        print(len(data[alg]), len(durations[alg]))
        t = durations[alg][i]
        rows.append([alg, score, t])
        result_dict[alg] = [score, t]
    table.add_rows(rows)
    if print_results:
        print(txt)
        print(table.draw())
    return result_dict
