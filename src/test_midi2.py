import os, numpy as np, pandas
np.random.seed(333)
os.chdir('src')
import mido  # , rtmidi, rtmidi_
import matplotlib.pyplot as plt

# local libs
import config
from data import data, midi, midi_generators as g
from utils import utils, io, plot
###

n = 10
multiTrack = True
context, x_train, labels = data.import_data(
    data.init(), n, multiTrack=multiTrack)
config.info('arrays2', x_train.shape)
# context, x_train, labels = data.import_data(data.init(), n, multiTrack=True)
# config.info('arrays2', x_train.shape)

dn = config.dataset_dir

# print(g.render())
# mid = g.render_midi(context, f=1)

# for m in mid:
#     print(m)

# io.export_midifile(mid, dn + 'cycle.mid')
# plot.multi(x_train[8, :30])

print('\n\n\n', '-MIDI-')
f = 5
# result = g.example(context)
# result = g.gen_data(context, 2, min_f=f, max_f=f)
result = g.gen_data_complex(
    context,
    1,
    min_f=f,
    max_f=f + 1,
    n_polyrythms=1,
    n_channels=3,
    d_phase=False,
    multiTrack=True)
config.info('arrays2', result.shape)
# print(' 000 ', result.shape, result[:10, :])
print(type(result))
print('result', result.shape)
# print(result[0, :5])

plot.multi(result[0, :30])

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
