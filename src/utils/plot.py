""" Midi-plots
Black indicates a note-on msg
Grey indicates a probable note-on msg (intensity correlates with p())
White indicates a rest
"""
import string
from utils import utils
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


def single(m, ylabels=pitches.all_keys):
    # set ylabels to [] to hide them
    if len(m.shape) > 2:
        m = m.reshape(m.shape[:-1])
    m = rotate_midi_matrix(m)
    # fig, ax = plt.subplots()
    plt.imshow(m, interpolation='nearest', cmap='gray_r')
    # fig.canvas.set_window_title(name + '...')
    # fig.set_title(name)
    # fig.set_xlabel('Time [iterations]')
    # fig.set_ylabel('Score')
    plt.yticks(
        np.arange(len(ylabels), 0, -1) - 1, ylabels, weight=1, size='xx-small')
    plt.show()
    return plt


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
    margin_y, margin_x = n * margin_top * 3, m * margin_left * 3
    figure = np.zeros((size1 * n + margin_y, size2 * m + margin_x))

    for i, yi in enumerate(sorted(x.keys())):
        for j, xi in enumerate(sorted(x[yi].keys())):
            x_decoded = x[yi][xi]
            sample = x_decoded[:size2].reshape((size2, size1))
            sample = rotate_midi_matrix(sample).reshape(size1, size2)
            # coordinates of the current sample
            a = i * size1 + i * margin_top * 3
            b = (i + 1) * size1 + i * margin_top * 3
            c = j * size2 + j * margin_left * 3
            d = (j + 1) * size2 + j * margin_left * 3
            # table separators (partially overlapping)
            border_color = 0.3
            figure[a, :] = 0
            figure[a + 1, 1:-1] = border_color
            figure[a + 2, :] = 0
            if vertical_borders:
                figure[:, c] = 0
                figure[1:, c + 1] = border_color
                figure[:, c + 2] = 0
                c, d = c + 3, d + 3
            a, b = a + 3, b + 3
            figure[a:b, c:d] = sample

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
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
    if min_y_scale:
        minn = min_y_scale
    else:
        minn = list(d.values())[0][0] - y_scale_margin

    if max_y_scale:
        maxx = max_y_scale
    else:
        maxx = minn + y_scale_margin

    std_min, std_max = 0, 0
    for k, v in d.items():
        if std:
            std_min = min(std[k])
            std_max = max(std[k])

        if min(v) - std_min < minn:
            minn = min(v) - std_min - y_scale_margin

        if max(v) + std_max > maxx:
            maxx = max(v) + std_max + y_scale_margin

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


def rotate_midi_matrix(m):
    return np.flip(m.copy().transpose(), axis=0)
