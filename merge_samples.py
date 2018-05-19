# author chinabluewu@github

import numpy as np
from tqdm import tqdm
import glob
#sudo pip install msgpack-python
import msgpack
import midi_manipulation
import os

#files = glob.glob('{}/*.mid*'.format(path))
try:
    path = os.getcwd()
    midi_dir = path + '\\gen'
    files = [os.path.join(midi_dir, f) for f in os.listdir(midi_dir) if f.endswith('mid')]
except Exception as e:
    raise e

songs = np.zeros((0,156))
for f in tqdm(files):
    try:
        song = np.array(midi_manipulation.midiToNoteStateMatrix(f))

        if np.array(song).shape[0] > 0:
            #songs.append(song)
            print(f)
            songs = np.concatenate((songs,song))
    except Exception as e:
        raise e
print ("samlpes merging ...")
midi_manipulation.noteStateMatrixToMidi(songs, "final")