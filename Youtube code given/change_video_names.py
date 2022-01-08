import os
import sys
import numpy as np

vid_dir = str(sys.argv[1])
old_vid_names = os.listdir(vid_dir)
new_vid_names = np.array(range(len(old_vid_names)))


mapping = []

for i in range(len(old_vid_names)):
    mapping.append([old_vid_names[i], str(new_vid_names[i]) + "." +old_vid_names[i].split('.')[-1]])

mapping = np.array(mapping)

for i in range(len(old_vid_names)):
    os.rename(os.path.join(vid_dir, old_vid_names[i]), os.path.join(vid_dir, mapping[i][1]))


np.save('vid_name_mapping.npy', mapping)

