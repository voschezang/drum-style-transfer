import re

NON_RYTHMS = ['fill', 'break']

# Filename conditions


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
