### from magenta.music.drums_encoder_decoder.py
# Default list of 9 drum types, where each type is represented by a list of
# MIDI pitches for drum sounds belonging to that type. This default list
# attempts to map all GM1 and GM2 drums onto a much smaller standard drum kit
# based on drum sound and function.

# Maschine/Ableton: BD = 24, SN/HH = 24, HH/SN = 25, OH = 26

BD = [36, 35]  # 24? == C1
SN = [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85]
HH = [42, 44, 54, 68, 69, 70, 71, 73, 78, 80]
OH = [46, 67, 72, 74, 79, 81]
T3 = [45, 29, 41, 61, 64, 84]
T2 = [48, 47, 60, 63, 77, 86, 87]
T1 = [50, 30, 43, 62, 76, 83]
CC = [49, 55, 57, 58]
RD = [51, 52, 53, 59, 82]
UNKNOWN = [127]
DRUMS = [BD, SN, HH, OH, T3, T2, T1, CC, RD, UNKNOWN]

keys = ['BD', 'SN', 'HH', 'OH', 'T3', 'T2', ' T1', ' CC', 'RD', 'UNKNOWN']
keys_full = [
    'Bassdrum', 'Snaredrum', 'Closed hi-hat', 'Open hi-hat', 'Floortom',
    'Low tom', 'High tom', ' Crash cymbal', 'Ride cymbal', 'Unknown'
]
all_keys = keys_full  # + ['Unknown']

# DEFAULT_DRUM_TYPE_PITCHES = [
#     # bass drum
#     [36, 35],

#     # snare drum
#     [38, 27, 28, 31, 32, 33, 34, 37, 39, 40, 56, 65, 66, 75, 85],

#     # closed hi-hat
#     [42, 44, 54, 68, 69, 70, 71, 73, 78, 80],

#     # open hi-hat
#     [46, 67, 72, 74, 79, 81],

#     # low tom
#     [45, 29, 41, 61, 64, 84],

#     # mid tom
#     [48, 47, 60, 63, 77, 86, 87],

#     # high tom
#     [50, 30, 43, 62, 76, 83],

#     # crash cymbal
#     [49, 55, 57, 58],

#     # ride cymbal
#     [51, 52, 53, 59, 82]
# ]


def used_note_list(drums, kit_size):
    result = []
    for note_list in drums:
        if kit_size == 1:
            result.append(note_list)
        else:
            for i in range(kit_size - 1):
                result.append([note_list[i]])
            result.append(note_list[i + 1:])

    return result
