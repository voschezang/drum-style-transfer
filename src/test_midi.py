import os, numpy as np, pandas
np.random.seed(333)
os.chdir('src')
import mido, rtmidi, rtmidi_

# local libs
import config, io
from data import data, midi
from utils import io

n = 2
context, x_train = data.init(n)

print(x_train.shape)

m = midi.decode(context, x_train[0])
print(type(m), m)

io.export_midifile(m)
