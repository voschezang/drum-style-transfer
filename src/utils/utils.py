""" A collection of generic functions that do not have their own modules
"""
import numpy as np, collections
import operator as op

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
