""" Project parameters & config
"""
tmp_model_dir = '/tmp/pattern-recognition_ml_models'  # see Makefile/logs
dataset_dir = '../datasets/'
plots_dir = 'results/'

# Verbose modes
result_ = True
debug_ = True


def debug(*args):
    if debug_:
        for a in args:
            print(a)
