import os, numpy as np, pandas
np.random.seed(333)
os.chdir('src')
import mido  # , rtmidi, rtmidi_

# local libs
import config
from data import data, midi, midi_generators as g
from utils import utils, io

###

n = 2
context, x_train, labels = data.import_data(data.init(), n)

dn = config.dataset_dir

# print(g.render())
mid = g.render_midi(context, f=1)

# for m in mid:
#     print(m)

# io.export_midifile(mid, dn + 'cycle.mid')

print(' no multi')
result = g.gen_data(context, 2)
result = g.gen_data_complex(context, 3)
print(type(result))
print(result.shape)

print('\n multi')
multi = False
result = g.gen_data_complex(context, 3, multiTrack=multi)
print(type(result))
print(result.shape)

# print(result[0, :15])
# print(result[0, :, 0].max())
# print(result[0, :, -1].max())
# print(utils.max_f(0.5), utils.max_f(1))
# print(utils.min_f(1), utils.min_f(10))

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
