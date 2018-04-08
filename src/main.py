# import rtmidi

# import time
# import rtmidi
# import rtmidi_

# from rtmidi_.midiconstants import NOTE_OFF, NOTE_ON

# print(NOTE_OFF)

# NOTE = 60  # middle C

# midiout = rtmidi.RtMidiOut()

# with (midiout.open_port(0) if midiout.get_ports() else
#       midiout.open_virtual_port("My virtual output")):
#     note_on = [NOTE_ON, NOTE, 112]
#     note_off = [NOTE_OFF, NOTE, 0]
#     midiout.send_message(note_on)
#     time.sleep(0.5)
#     midiout.send_message(note_off)

# del midiout

import mido

# port = mido.open_output()
# port.send(msg)

mid = mido.MidiFile('mary.mid')
# for msg in mid.play():
# port.send(msg)
