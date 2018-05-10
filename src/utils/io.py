import os, re, time, datetime, pandas, numpy as np, collections
import mido

import config
from data import data, midi


def import_mididata(c, dirname='../datasets/examples/', n=2, r=False):
    # c :: data.Context
    if not dirname[-1] == '/': dirname += '/'
    filenames = search(dirname, n, is_midifile, r)
    midis = []
    for fn in filenames:
        midis.append(import_midifile(fn))
    return midis, filenames


def search(dirname, max_n, add_cond, r=False):
    if r:
        return walk_and_search(dirname, add_cond, max_n)

    files = os.listdir(dirname)
    return [dirname + fn for fn in files if add_cond(fn)][:max_n]


def walk_and_search(dirname, add_cond, max_n=100):
    # return a list of filenames that are present (recursively) in 'dirname' and
    # satisfy 'add_cond'
    n = 0
    result = []
    for path, dirs, filenames in os.walk(dirname):
        for fn in filenames:
            if add_cond(fn):
                result.append(path + '/' + fn)
                n += 1
        if n >= max_n:
            return result[:max_n]
    return result


def import_midifile(filename='../mary.mid'):
    if not filename[-4:] == '.mid':
        filename += '.mid'
    config.info('reading file: %s' % filename)
    return mido.MidiFile(filename)


def export_midifile(mid, filename='../song_export.mid'):
    if not filename[-4:] == '.mid':
        filename += '.mid'
    mid.save(filename)


def is_midifile(fn: str) -> bool:
    return fn[-4:] == '.mid'


###
###
###


def save_to_csv(dirname, name, data):
    # panda df requires data to be NOT of type {key: scalar}
    # but rather: {'name':['value']}
    filename = dirname + "/" + name + ".csv"
    df = pandas.DataFrame(data=data)
    df.to_csv(filename, sep=',', index=False)
    # mkdir filename
    # for k in d.keys(): gen png
    return filename


def print_dict(dirname="", d={}, name="text"):
    if not dirname == "":
        dirname += "/"
    name += ".txt"
    with open(dirname + "0_" + name, "w") as text_file:
        print(name + "\n", file=text_file)
        for k, v in d.items():
            # print(f"{k}:{v}", file=text_file) # pythonw, python3
            print('{:s}, {:s}'.format(str(k), str(v)), file=text_file)


def replace_special_chars(string, char='-'):
    return re.sub('[^A-Za-z0-9]+', char, string)


def make_dir(name="results_abc", post='', timestamp=False):
    name = replace_special_chars(name)
    post = replace_special_chars(post)
    if not os.path.isdir(name):
        os.mkdir(name)
    if timestamp:
        dirname = name + "/" + shortTimeStamp() + ' ' + post
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname
    return name


def make_subdir(parent, name="img"):
    name = parent + "/" + name
    if not os.path.isdir(parent):
        print("error parent")
        os.mkdir(parent)
    if not os.path.isdir(name):
        os.mkdir(name)
    return name


def shortTimeStamp(s=5):
    n = 1000 * 1000  # larger n -> larger timestamp
    t = time.time()
    sub = int(t / n)
    t = str(round(t - sub * n, s))
    date = datetime.datetime.date(datetime.datetime.now())
    return str(date) + "_" + str(t)


def unique_dir(name, post=''):
    # name = 'iterative' | 'constructive'
    # generate unique dir
    dirname = make_dir('results-' + name, timestamp=True, post=post)
    img_dir = make_subdir(dirname, 'img')
    return dirname, img_dir
