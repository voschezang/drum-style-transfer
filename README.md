# Drum style transfer

This repository is part of this [research paper](https://link.springer.com/chapter/10.1007%2F978-3-030-31978-6_10) and revolves around style transfer of MIDI drum patterns.
The main features are:
- Training a model on an exsiting dataset (see links below)
- Use a model to generate new drum patterns
- Analyze the similarity of groups of patterns

This not meant as a standalone application but with a bit of tweaking you should be able to use the code to train your own models.

Listen to `sample03.wav` and `sample12.wav` to get an impression for the type music that can be generated. 
These patterns were produced by decoding a random walk through the 10-dimensional latent space.


<hr/>

List of datasets datasets [link](https://github.com/midi-ld/machine-readable-datasets).

Direct links
 - Lakh MIDI dataset, partly matched to the Million Song Dataset.
 - [ "The largest midi collection on the internet" ](https://www.reddit.com/r/WeAreTheMusicMakers/comments/3anwu8/the_drum_percussion_midi_archive_800k/)

This project was inspired by [Magenta's MusicVAE](https://magenta.tensorflow.org/music-vae). Here is a [demo](https://experiments.withgoogle.com/ai/beat-blender/view/) of an application based on this network.


## Setup

Clone repo and install dependencies (use `sudo` on linux).
```
git clone https://github.com/voschezang/drum-style-transfer
cd drum-style-transfer
make deps
```

Make sure that 'data_format' your keras config file (`~/.keras/keras.json`) is set to 'channels_last'. Otherwise it should be changed manually in every script/notebook

## Usage

Start jupyter. This should open `http://localhost:8888/` in your default webbrowser.
```
make start
```
There are a number of notebooks. These show how the functions (e.g. in `src/models.py`) can be used.
Depending on your usecase you'll have to look at different notebooks.

The folder `ableton` contains an [Ableton](https://www.ableton.com/en/) project that can be used to synthesize MIDI files (i.e. generate audiofiles). The project may display some errors about missing (fx-)plugins but these can be ingored.

<br/>
<hr/>

## About

The project uses [mypy](https://github.com/python/mypy) type definitions _(e.g. f(x:int)-> int)_. However, due to issues with unsupported (external) modules, type checking is not implemented. Types are to be used solely for documentation purposes.
