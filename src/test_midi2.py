import os, numpy as np
if __name__ == "__main__":
    np.random.seed(333)
    print(os.getcwd())

import mido  # , rtmidi, rtmidi_
import matplotlib.pyplot as plt

import config
import setup
import midi
import midi.decode
from midi import generators as g
from utils import utils, io, plot

if __name__ == "__main__":
    context = setup.init()
    n = 10
    multiTrack = True
    reduce_dims = midi.ReduceDimsOptions.GLOBAL
    reduce_dims = midi.ReduceDimsOptions.MIDIFILE
    dim4 = True
    dirname = 'drum_midi'
    x_train, labels = setup.import_data(
        context, n, multiTrack, reduce_dims, dim4, dirname, r=True)
    config.info('x_train', x_train.shape)
    # context, x_train, labels = data.import_data(data.init(), n, multiTrack=True)
    # config.info('arrays2', x_train.shape)

    # dn = config.dataset_dir
    # io.export_midifile(mid, dn + 'cycle.mid')
    print(labels[0])
    plot.single(x_train[0, :80, :, 0])

    print('\n\n\n-MIDI-\n')
    f = 5
    # result = g.example(context)
    # result = g.gen_data(context, 2, min_f=f, max_f=f)
    result, params = g.gen_data_complex(
        context,
        1,
        min_f=f,
        max_f=f + 1,
        n_polyrythms=1,
        n_channels=3,
        d_phase=False,
        multiTrack=True)
    config.info('result', result.shape)
    # print(' 000 ', result.shape, result[:10, :])
    print(type(result))
    print('result', result.shape)
    # print(result[0, :5])

    a = midi.MultiTrack.from_array(result[0])
    print(type(a))

    # midi.decode.track(context, result[0])
    # plot.single(result[0, :30])

# fn = dn + '4-floor-120bpm.mid'
# mid = io.import_midifile(fn)

# mid = io.import_midifile(dn + 'song_export.mid')
# mid = io.import_midifile(dn + 'examples/01 8th Cym.mid')

# encoded = midi.encode(context, mid)
# # encoded = x_train[0]
# decoded = midi.decode_track(context, encoded)

# m = decoded
# print(type(m), m, encoded.shape)
# print('1000 tick2second',
#       mido.tick2second(context.n_instances, context.ticks_per_beat,
#                        context.tempo))
# print('second2tick',
#       mido.second2tick(context.max_t, context.ticks_per_beat, context.tempo))
# print(mid.length, m.length, context.max_t)

# # for x in mid: print(x)
# print(mid.filename)
# io.export_midifile(m, dn + 'song_export_copy.mid')

# print(m, m.tracks[0])
# print(encoded.shape)
