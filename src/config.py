""" Project parameters & config
- dirnames
- verbose modes

This module should not have any dependencies
"""
tmp_model_dir = '/tmp/pattern-recognition_ml_models'  # !important: see makefile/LOG_DIR
dataset_dir = '../datasets/'
plots_dir = 'results/'

# Verbose modes
result_ = True
debug_ = True
info_ = True


def debug(*args):
    if debug_:
        print('DEBUG --- --- ---')
        for a in args:
            print(a)


def info(*args):
    if info_:
        print(' \________ ')
        print(' | [INFO] \ __ __ __')
        for a in args:
            print(' |  ', a)
        print(' \ \n  |\n /')
