# Pattern recognition

List of datasets datasets [link](https://github.com/midi-ld/machine-readable-datasets).

Direct links
 - Lakh MIDI dataset, partly matched to the Million Song Dataset.
 - [ "The largest midi collection on the internet" ](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3anwu8/the_drum_percussion_midi_archive_800k/)

This project is inspired by [Magenta's MusicVAE](https://github.com/tensorflow/magenta)


## Setup

Install dependencies (use `sudo` on linux).
```
make deps
```

Make sure that 'data_format' your keras config file (`~/.keras/keras.json`) is set to 'channels_last'. Otherwise it should be changed manually in every script/notebook


## About

The project uses [mypy](https://github.com/python/mypy) type definitions _(e.g. f(x:int)-> int)_. However, due to issues with unsupported (external) modules, type checking is not implemented. Types are to be used solely for documentation purposes.
