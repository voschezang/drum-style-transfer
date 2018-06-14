import os, re, time, datetime, pandas, numpy as np, collections
import mido, bz2

import config
from utils import string

IGNORE_DIRS = [
    'Analogue Drums',
    'Asia',
    'Cha cha',
    'Drum Patterns',
    'Drumatic Beats',
    'Electronic Dance',
    'Ending',
    'Europe',
    'Fills Unlimited',
    'GM - AC Percussion',
    'GM - World Beats SSD',
    'GM - World Beats Superior',
    'Cowbell',
    'Midi.Styles.Percussion',
    'Hi-Hat & Noise Loops',
    'L.A.Riot.Drum.Loops',
    'Cinematic - 4 Bars',
    'Cinematic - 8 Bars',
    'DrumaticBeats_MIDIDrumLoops',
]  # (case sensitive)


def reset_tmp_dir():
    os.system('rm -r ' + config.tmp_dir)
    os.system('mkdir ' + config.tmp_dir)
    return True


def save(obj, fn):
    if not fn[:-4] == '.pkl':
        fn += '.pkl'
    with open(fn, 'wb') as f:
        pickle.dump(transformations, f)
    return fn


def load(obj, fn='obj.pkl'):
    with open(fn, 'rb') as file:
        x = pickle.load(file)
    return x


def save_dict(dn, name, data={'k': ['v']}):
    # panda df requires data to be NOT of type {key: scalar}
    # but rather: {'name':['value']}
    if len(dn) > 0 and not dn[-1] == '/':
        dn += '/'
    if not name[-4:] == '.csv':
        name += '.csv'
    fn = dn + name
    df = pandas.DataFrame(data=data)
    df.to_csv(fn, sep=',', index=False)
    return fn


def read_dict(fn):
    if not fn[-4] == '.':
        fn += '.csv'
    return pandas.read_csv(fn).to_dict()


def read_dict_dir(dn='dir/'):
    """e.g.
    dir/
      file1.csv
      file2.csv

    result = {'dir1': {'file1': []}}
    """
    result = {}
    for fn in os.listdir(dn):
        # ignore non-csv files
        if fn[-4:] == '.csv':
            key = fn[:-4]  # rm extension .csv
            result[key] = read_dict(dn + fn)
    return result


def read_categorical_dict_dir(dn='dir/'):
    """e.g.
    dir/
      class1/
        subclass1.csv
      class2/
        subclass2.csv

    result = {'class/sub_class': {}}
    """
    result = {}
    print(os.listdir(dn))
    for sub_dir in os.listdir(dn):
        sub_dir += '/'
        if os.path.isdir(dn + sub_dir):
            print('sd', sub_dir)
            for fn in os.listdir(dn + sub_dir):
                # ignore non-csv files
                if fn[-4:] == '.csv':
                    print('csv', fn)
                    key = fn[:-4]  # rm extension .csv
                    result[sub_dir + key] = read_dict(dn + sub_dir + fn)
    return result


def import_mididata(c,
                    dirname='../datasets/examples/',
                    n=2,
                    cond=string.is_midifile,
                    r=False):
    # c :: data.Context
    if not dirname[-1] == '/':
        dirname += '/'
    filenames = search(dirname, n, cond, r)
    midis = []
    for fn in filenames:
        mid = import_midifile(fn)
        if mid.tracks[0]:
            midis.append(mid)
    return midis, filenames


def search(dirname, max_n, add_cond, r=False):
    if r:
        return walk_and_search(dirname, add_cond, max_n)

    files = os.listdir(dirname)
    return [dirname + fn for fn in files if add_cond(fn)][:max_n]


def walk_and_search(dirname, add_cond, max_n=100):
    # return a list of filenames that are present (recursively) in 'dirname'
    # and satisfy 'add_cond'
    print('walk_and_search(%s)' % dirname)
    n = 0
    result = []
    for path, _dirs, filenames in os.walk(dirname):
        print('path', path)
        if not ignore_path(path):
            for fn in filenames:
                if add_cond(fn):
                    result.append(path + '/' + fn)
                    print(' ', fn)
                    n += 1
            if n >= max_n:
                return result[:max_n]
        else:
            print('path ignored: %s \n' % path)
    return result


def ignore_path(path='foo/bar'):
    for name in IGNORE_DIRS:
        if name in path.split('/'):
            config.info('path ignored: name `%s` in IGNORE_DIRS' % name)
            return True
    return False


def import_midifile(fn='../mary.mid', convert=True):
    if not fn[-4:] == '.mid':
        fn += '.mid'
    if convert:
        return mido.MidiFile(fn)
    with open(fn, encoding='latin-1') as mid:
        data = mid.read().encode('utf-8')
    return data


def export_midifile(mid, fn='../song_export.mid'):
    if not fn[-4:] == '.mid':
        fn += '.mid'
    mid.save(fn)
    return fn


def export_MultiTrack(data, fn='track'):
    # TODO numpy.savez
    if len(data.shape) == 3:
        data = data[:, :, 0]
    if not fn[-4:] == '.csv':
        fn += '.csv'
    np.savetxt(fn, data, delimiter=',', fmt='%.3f')  #, fmt='%.4e'
    return fn


def compress_MultiTrack(x, tmp_name='original'):
    fn = export_MultiTrack(x, config.tmp_dir + tmp_name)
    with open(fn) as a:
        raw = a.read().encode('utf-8')
    return bz2.compress(raw)


###
###
###

# def print_dict(dirname="", d={}, name="text"):
#     if not dirname == "":
#         dirname += "/"
#     name += ".txt"
#     with open(dirname + "0_" + name, "w") as text_file:
#         print(name + "\n", file=text_file)
#         for k, v in d.items():
#             # print(f"{k}:{v}", file=text_file) # pythonw, python3
#             print('{:s}, {:s}'.format(str(k), str(v)), file=text_file)

# def replace_special_chars(string, char='-'):
#     return re.sub('[^A-Za-z0-9]+', char, string)

# def make_dir(name="results_abc", post='', timestamp=False):
#     name = replace_special_chars(name)
#     post = replace_special_chars(post)
#     if not os.path.isdir(name):
#         os.mkdir(name)
#     if timestamp:
#         dirname = name + "/" + shortTimeStamp() + ' ' + post
#         if not os.path.isdir(dirname):
#             os.mkdir(dirname)
#         return dirname
#     return name

# def make_subdir(parent, name="img"):
#     name = parent + "/" + name
#     if not os.path.isdir(parent):
#         print("error parent")
#         os.mkdir(parent)
#     if not os.path.isdir(name):
#         os.mkdir(name)
#     return name

# def shortTimeStamp(s=5):
#     n = 1000 * 1000  # larger n -> larger timestamp
#     t = time.time()
#     sub = int(t / n)
#     t = str(round(t - sub * n, s))
#     date = datetime.datetime.date(datetime.datetime.now())
#     return str(date) + "_" + str(t)

# def unique_dir(name, post=''):
#     # name = 'iterative' | 'constructive'
#     # generate unique dir
#     dirname = make_dir('results-' + name, timestamp=True, post=post)
#     img_dir = make_subdir(dirname, 'img')
#     return dirname, img_dir
