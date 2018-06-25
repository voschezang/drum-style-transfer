""" Midi-plots
Black indicates a note-on msg
Grey indicates a probable note-on msg (intensity correlates with p())
White indicates a rest
"""
from utils import utils, string
from midi import pitches

from scipy.stats import norm
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [
    'Times New Roman', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana'
]
rcParams['font.size'] = 14
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# These are the "Tableau 20" colors as RGB.
TABLEAU20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199,
                                                                 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(TABLEAU20)):
    r, g, b = TABLEAU20[i]
    TABLEAU20[i] = (r / 255., g / 255., b / 255.)

### --------------------------------------------------------------------
### Plot functions
### --------------------------------------------------------------------


def single(m, ylabels=pitches.all_keys, figsize=(10, 10), fn=None):
    # set ylabels to [] to hide them
    if len(m.shape) > 2:
        m = m.reshape(m.shape[:-1])
    xlength = m.shape[0]
    m = _rotate_midi_matrix(m)
    fig = plt.figure(figsize=figsize)
    _midi_grid(fig, ylabels, n_bars=1, length=xlength)
    plt.imshow(m, interpolation='nearest', cmap='gray_r')
    plt.show()
    if not fn is None:
        fn = string.to_dirname(fn)
        plt.savefig(fn + '-plot.png')


def multi(x, crop_size=40, margin_top=1, margin_left=1, v=0):
    """
    x :: {x_pos: {y_pos: np.ndarray}} | np.ndarray
    """
    if isinstance(x, np.ndarray):
        if v: print('converting x to double-dict')
        x_ = {}
        for i in range(x.shape[0]):
            x_[i] = {0: x[i]}
        x = x_

    n = len(x.keys())
    m = len(utils.get(x)[-1].keys())
    vertical_borders = not m == 1

    # display a 2D manifold of output samples
    x_sample = utils.get(x, 1)[-1]
    size1 = x_sample.shape[1]
    size2 = crop_size  # crop x_train.shape[1]
    margin_y, margin_x = n * margin_top * 1, m * margin_left * 1
    figure = np.zeros((size1 * n + margin_y, size2 * m + margin_x))
    ylabels = []
    for i, yi in enumerate(sorted(x.keys())):
        ylabels += pitches.all_keys + ['']
        for j, xi in enumerate(sorted(x[yi].keys())):
            x_decoded = x[yi][xi]
            sample = x_decoded[:size2].reshape((size2, size1))
            sample = _rotate_midi_matrix(sample).reshape(size1, size2)
            # coordinates of the current sample
            a = i * size1 + i * margin_top
            b = (i + 1) * size1 + i * margin_top
            c = j * size2 + j * margin_left
            d = (j + 1) * size2 + j * margin_left
            c, d = c + 1, d + 1
            a, b = a + 1, b + 1
            figure[a:b, c:d] = sample

    fig = plt.figure(figsize=(10, 10))
    _midi_grid(fig, ylabels, n_bars=crop_size / 40, length=x_sample.shape[0])
    plt.imshow(figure, cmap='gray_r')
    plt.show()


def line(matrix):
    plt.plot(matrix[:30, 0])
    plt.show()


def custom(d,
           title='',
           options={},
           log=False,
           min_y_scale=0.,
           max_y_scale=1.,
           y_scale_margin=0.1,
           type_='line',
           std={},
           figsize=(6, 3),
           dn=None,
           show=False):
    """
    d :: {label: [value]}
    if dn:
      save fig to `[dn]/dict_plot-[title]`

    options.keys = { x_labels :: []
        , x_ticks :: []
        , y_labels :: []
        , y_label :: ''
        , legend :: bool
        , title :: bool
    }
    Maybe x :: None | x

    """
    plt.figure(figsize=figsize)
    name = title
    labels = []
    for s in d.keys():
        labels.append(s)
    n_labels = len(labels)

    if type_ == 'line':
        plots = plot_dict(d)
    elif type_ == 'bar':
        plots = bar_plot(d, std)
    else:
        print('WARNING unkown arg value: `type_` was %s' % type_)

    # set range of y axis
    minn = list(d.values())[0][0] - y_scale_margin
    maxx = minn + y_scale_margin
    std_ = 0
    for k, v in d.items():
        if std:
            std_ = max(std[k])
        if min(v) - std_ <= minn + y_scale_margin:
            minn = min(v) - std_ - y_scale_margin
        if max(v) + std_ >= maxx - y_scale_margin:
            maxx = max(v) + std_ + y_scale_margin

    if not max_y_scale is None:
        maxx = max_y_scale
    if not min_y_scale is None:
        minn = min_y_scale

    plt.ylim([minn, maxx])

    # logaritmic y axis
    if log:
        plt.yscale('symlog')

    # title, legend, labels

    if 'legend' in options.keys():
        handles = [
            mpatches.Patch(color=TABLEAU20[i], label=labels[i])
            for i in range(0, n_labels)
        ]
        # legend inside subplot
        plt.legend(loc=4, handles=handles)
    # legend on top of subplot
    # plt.legend(
    #     handles=handles,
    #     bbox_to_anchor=(0., 1.02, 1., .102),
    #     loc=3,
    #     ncol=2,
    #     mode="expand",
    #     borderaxespad=0.)

    plt.title(name)
    if 'y_label' in options.keys():
        plt.ylabel(options['y_label'])

    if 'x_labels' in options.keys():
        x_labels = options['x_labels']
        plt.xticks(range(len(x_labels)), x_labels)

    # plt.text(50, 12, "lorem ipsum", fontsize=17, ha="center")
    if log:
        name += '-log'
    if dn:
        dn = string.to_dirname(dn)
        plt.savefig(dn + 'dict_plot-' + name + '.png')

    if not show:
        plt.clf()
        plt.close()


def bar_plot(d={}, std={}):
    """e.g.
    d = {'male': [], 'female':[]}
    std = {'male': [], 'female':[]}
    """
    plots = []
    for k, v in d.items():
        ind = np.arange(len(v))
        width = 0.35
        if std:
            p1 = plt.bar(ind, v, width, yerr=std[k])
        else:
            p1 = plt.bar(ind, v, width)
        plots.append(p1)
    return plots


def plot_dict(d, minn=0, maxx=1):
    """ plot.dict()
    d :: {label: [value]}
    if dn:
      save fig to `[dn]/dict_plot-[title]`

    options.keys = {x_labels, x_ticks, y_labels, y_label, legend}

    """
    plots = []
    for i, (k, v) in enumerate(d.items()):
        v = np.array(v)
        # y axis limits
        p1 = plt.plot(v, lw=2, color=TABLEAU20[i])
        plots.append(p1)
    return plots


def _rotate_midi_matrix(m):
    return np.flip(m.copy().transpose(), axis=0)


def _midi_xticks(ax, n_bars=1, length=40, d=10):
    # d = resolution
    n_beats = 4  # beats per bar (4/4 time signature)
    n_bars = int(np.floor(n_bars))
    n_extra_beats = int((length / d) % 4)
    # sub_labels = np.arange(n_beats) + 1
    labels_full_bars = list(np.arange(n_beats) + 1) * n_bars
    labels_extra_beats = list(np.arange(n_extra_beats) + 1)
    labels = labels_full_bars + labels_extra_beats
    for bar in range(n_bars):
        for beat in range(n_beats):
            labels[bar * n_beats + beat] = str(bar + 1) + ':' + str(
                labels[bar * n_beats + beat])
    bar += 1
    for beat in range(n_extra_beats):
        labels[bar * n_beats + beat] = str(bar + 1) + ':' + str(
            labels[bar * n_beats + beat])

    # labels = [i + ':' + sub_label[i] for i in range(n)]
    n_total_beats = n_bars * n_beats + n_extra_beats
    major_ticks = np.arange(0, n_total_beats * d + 1, 40)
    minor_ticks = np.arange(0, n_total_beats * d + 1, 5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    plt.xticks(np.arange(0, d * n_total_beats, d), labels)


def _midi_yticks(ax, ylabels=[]):
    n = len(ylabels)
    major_ticks = np.arange(0, n + 1, 1)
    minor_ticks = np.arange(0, n + 1, 10 + 1)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    plt.yticks(
        np.arange(len(ylabels), 0, -1) - 1, ylabels, weight=1, size='x-small')


def _midi_grid(fig, ylabels=[], n_bars=1, length=50):
    ax = fig.add_subplot(1, 1, 1)
    _midi_yticks(ax, ylabels)
    _midi_xticks(ax, n_bars, length)
    ax.grid(which='minor', alpha=0.3, linewidth=2)
    ax.grid(which='major', alpha=0.5)
