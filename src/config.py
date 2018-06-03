""" Project parameters & config
- dirnames
- verbose modes

This module should not have any dependencies
"""
project_name = 'pattern-recognition'

tmp_log_dir = '/tmp/' + project_name + '_ml_models'  # !important: see makefile/LOG_DIR
dataset_dir = '../datasets/'
model_dir = dataset_dir + 'models/'
export_dir = '../midis/'
plots_dir = 'results/'
tmp_dir = '../tmp/'

seed = 377


class Colors:
    default = '\033[0m'
    green = '\033[92m'
    red = '\033[91m'


# Verbose modes
result_ = True
debug_ = True
info_ = True


def debug(*args):
    if debug_:
        print(Colors.red, '[DEBUG] >', Colors.default)
        for a in args:
            print(' |> ', a)


def info(*args):
    if info_:
        print(Colors.green, '[INFO] :', Colors.default)
        for a in args:
            print(' | ', a)
