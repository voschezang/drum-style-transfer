import os, numpy as np, pandas
np.random.seed(333)
os.chdir('src')
import mido, rtmidi, rtmidi_

# local libs
import config
from data import data, midi
from utils import io

n: int = 2
context, x_train = data.init(n)
print(x_train.shape)

dn = config.dataset_dir
mid = io.import_midifile()
encoded = midi.encode(context, mid)
decoded = midi.decode_track(context, encoded)

# # mid = io.import_midifile(dn + 'song_export.mid')
# mid = io.import_midifile(dn + 'examples/01 8th Cym.mid')

# # encoded = midi.encode(context, mid)
# encoded = x_train[0]
# decoded = midi.decode(context, encoded)
m = decoded
print(type(m), m, encoded.shape)
print('1000 tick2second',
      mido.tick2second(context.n_instances, context.ticks_per_beat,
                       context.tempo))
print('second2tick',
      mido.second2tick(context.max_t, context.ticks_per_beat, context.tempo))
print(mid.length, m.length, context.max_t)

# for x in mid: print(x)

# io.export_midifile(m, dn + 'song_export_copy.mid')
