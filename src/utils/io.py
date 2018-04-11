import os, pandas, numpy as np, collections
import mido

import config
from data import data, midi


def import_data(c, dirname, n=2):
    # c :: data.Context
    files = os.listdir(dirname)
    filenames = [f for f in files if not f == '.DS_Store'][:n]
    # filenames = os.listdir(dirname)[:n]

    midis = []
    for fn in filenames:
        print('reading file: %s' % fn)
        mid = mido.MidiFile(dirname + fn)
        midis.append(mid)
    return midis


def export_midifile(mid, filename='song_export.mid'):
    mid.save(filename)


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
