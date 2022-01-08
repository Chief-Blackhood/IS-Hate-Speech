import os
import sys
import subprocess
import numpy as np
import pysrt

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

srt_files = sys.argv[1]
vid_shots_dir = sys.argv[2]
per_shot_srt_file_dir = sys.argv[3]

for vid in os.listdir(vid_shots_dir):
    if os.path.exists(os.path.join(srt_files, vid + '.srt')):
        os.mkdir(os.path.join(per_shot_srt_file_dir, vid))
        shots = os.listdir(os.path.join(vid_shots_dir, vid))
        shots.sort(key=lambda f: int(f.split(".")[0]))
        t0 = 0
        t1 = 0
        for shot in shots:
            # print(t0)
            # print(t1)
            t0 = t1
            t1 = t0 + get_length(os.path.join(vid_shots_dir, vid, shot))
            subs = pysrt.open(os.path.join(srt_files, vid+'.srt'))
            shot_subs = []
            for i in subs:
                st = i.start.minutes*60 + i.start.seconds + i.start.milliseconds/1000
                et = i.end.minutes*60 + i.end.seconds + i.end.milliseconds/1000
                if t0 <= et:
                    if t1 >= st:
                        shot_subs.append(i.text)
                    else:
                        break

            temp_shot_sub = [j for s in shot_subs for j in s.strip().split('\n')]
            final_shot_sub = []
            for i in temp_shot_sub:
                if i not in final_shot_sub:
                    final_shot_sub.append(i)

            out = '\n'.join(final_shot_sub)
            with open(os.path.join(per_shot_srt_file_dir, vid, shot[:-4] + '.txt'), 'w+') as f:
                f.write(out)
                f.close()
        print(vid, " done")
    else:
        print(vid, "srt file missing")