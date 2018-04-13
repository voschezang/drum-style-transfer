import os, numpy as np, pandas
np.random.seed(333)
os.chdir('src')
import mido, rtmidi, rtmidi_

# local libs
import config, io
from data import data, midi
from utils import io

n: int = 2
context, x_train = data.init(n)
print(x_train.shape)

dn = config.dataset_dir
# mid = io.import_midifile(dn + 'song_export.mid')
mid = io.import_midifile(dn + 'examples/01 8th Cym.mid')

# encoded = midi.encode(context, mid)
encoded = x_train[0]
decoded = midi.decode(context, encoded)
m = decoded
print(type(m), m)
print(mid.length, m.length)

io.export_midifile(m, dn + 'song_export_copy.mid')
