def decode_track(c, matrix: MultiTrack) -> mido.MidiTrack:
    # c :: data.Context
    # matrix :: [ vector per instance ]
    # vector :: [ notes ]

    if not isinstance(matrix, MultiTrack):
        config.debug('decode_track - input was not MultiTrack.',
                     'Assuming MultiTrack')
    # if isinstance(matrix, Track):
    #     multi = False
    # elif isinstance(matrix, MultiTrack):
    #     multi = True
    # else:
    #     config.debug('decode_track - input was not Track | MultiTrack.',
    #                  'Assuming MultiTrack')
    #     multi = True

    # decode notes for each instance
    track = mido.MidiTrack()
    # msgs = []
    t = 0
    for i, vector in enumerate(matrix):
        # lookahead_matrix = the part of the matrix that occurred within
        # 'PADDING' cells before 'i'
        lookahead_matrix = matrix[i - PADDING:i]
        # msgs :: mido.Message, with absolute t in seconds
        msgs = decode_notes(c, Notes(vector), t, lookahead_matrix)
        track.extend(msgs)
        t += c.dt

    # convert absolute time in seconds to relative ticks
    track.sort(key=lambda msg: msg.time)
    track = convert_time_to_relative_value(track, lambda t: second2tick(c, t))

    mid = mido.MidiFile()
    mid.ticks_per_beat = c.ticks_per_beat
    mid.tracks.append(track)
    config.info('len, max_t', mid.length, c.max_t)
    return mid


def notes(c, notes: Notes, t, lookahead_matrix=None) -> List[mido.Message]:
    # :t :: seconds
    # msg.time = absolute, in seconds
    if not isinstance(notes, Notes):  # np.generic
        errors.typeError('numpy.ndarray', notes)
    msgs = []
    for note_index, velocity in enumerate(notes):
        if lookahead_matrix is None or lookahead_matrix[:, note_index].max(
        ) < MIDI_NOISE_FLOOR:
            msgs.extend(decode_note(c, note_index, velocity, t))
    return msgs


def note(c, note_index, velocity, t):
    # return ::  [] | a list of midi messages (note on, note off)
    if velocity < MIDI_NOISE_FLOOR:
        return []
    if note_index < SILENT_NOTES:
        return []
    # Convert note_index in array to actual note-value
    note = LOWEST_NOTE + note_index - SILENT_NOTES
    if note > HIGHEST_NOTE:
        config.debug('decode_note: note index > highest note')
    return gen_note_on_off(c, note_index, 127, t)
