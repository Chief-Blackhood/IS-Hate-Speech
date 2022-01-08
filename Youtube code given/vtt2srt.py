import os
import sys
import numpy as np

transcript_dir = sys.argv[1]
vid_name_mapping = np.load(sys.argv[2])
print(vid_name_mapping)

transcript_files = [[i[0].replace(i[0].split('.')[-1], '') + 'en.vtt', i[1].split('.')[0] + '.srt'] for i in vid_name_mapping]

print("\n\n\n\n\n"+str(transcript_files))


for i in transcript_files:
    if os.path.exists(os.path.join(transcript_dir, i[0])):
        print("here")
        os.rename(os.path.join(transcript_dir, i[0]), os.path.join(transcript_dir, i[1].split('.')[0]+'.vtt'))
        os.system("ffmpeg -i " + os.path.join(transcript_dir, i[1].split('.')[0]+'.vtt') + " " + os.path.join(transcript_dir, i[1]))
