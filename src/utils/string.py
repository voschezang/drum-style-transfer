import re

NON_RYTHMS = ['fill', 'break']

# Filename conditions


def to_dirname(dn='dir'):
    if not dn[-1] == '/':
        return dn + '/'
    return dn


def end_with(fn='file', suffix='.csv'):
    if not fn[-len(suffix):] == suffix:
        return fn + suffix
    return fn


def is_drumrythm(fn: str) -> bool:
    tokens = tokenize(fn)
    for token in tokens:
        if token in NON_RYTHMS:
            return False
    return True and is_midifile(fn)


def is_midifile(fn: str) -> bool:
    return fn[-4:] == '.mid'


# Generic


def stem(string='abc.jpg'):
    # rmv file extension (.jpg)
    return string.split('.')[0]


def replace_special_chars(string, replacement=''):
    # keep spaces
    return re.sub('[^A-Za-z0-9]+', replacement, string)


def tokenize(string='01 Song'):
    separators = [' ', '_', '-', '.', ':'] + [str(i) for i in range(10)]
    tokens = [string.lower()]
    for s in separators:
        ls = [token.split(s) for token in tokens]
        tokens = []
        for items in ls:
            tokens.extend(items)
    return tokens


def extract_labels_from_filename(fn='class/subclass/filename'):
    # return list of (genre, subgenre)
    items = fn.split('/')
    if len(items) > 1:
        return items[:-1]
    return items
