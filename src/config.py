""" Project parameters & config
- dirnames
- verbose modes

This module should not have any dependencies
"""
tmp_logs_dir = '/tmp/pattern-recognition_ml_models'  # !important: see makefile/LOG_DIR
dataset_dir = '../datasets/'
model_dir = dataset_dir + 'models/'
export_dir = '../midis/'
plots_dir = 'results/'

seed = 377

# Verbose modes
result_ = True
debug_ = True
info_ = True


def debug(*args):
    if debug_:
        print('[DEBUG] >')
        for a in args:
            print(' |> ', a)


def info(*args):
    if info_:
        print('[INFO] :')
        for a in args:
            print(' | ', a)
