import cv2
import os
import sys

import time
import subprocess

import subprocess

def extract_shots_with_ffprobe(src_video, threshold=0.3):
    """
    uses ffprobe to produce a list of shot 
    boundaries (in seconds)
    
    Args:
        src_video (string): the path to the source 
            video
        threshold (float): the minimum value used 
            by ffprobe to classify a shot boundary
    
    Returns: 
        List[(float, float)]: a list of tuples of floats 
        representing predicted shot boundaries (in seconds) and 
        their associated scores
    """
    scene_ps = subprocess.Popen(("ffprobe", 
                                "-show_frames",
                                "-of",
                                "compact=p=0", 
                                "-f",
                                "lavfi",
                                "movie=" + src_video + ",select=gt(scene\," + str(threshold) + ")"),
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.STDOUT)
    output = scene_ps.stdout.read()
    print("######OUTPUT FROM SHOT EXTRACTION ######\n"+str(output)+"\n##############################################\n")
    boundaries = extract_boundaries_from_ffprobe_output(output.decode())
    return boundaries

def extract_boundaries_from_ffprobe_output(output):
    """
    extracts the shot boundaries from the string output
    producted by ffprobe
    
    Args:
        output (string): the full output of the ffprobe
            shot detector as a single string
    
    Returns: 
        List[(float, float)]: a list of tuples of floats 
        representing predicted shot boundaries (in seconds) and 
        their associated scores
    """
    boundaries = []
    for line in output.split('\n'):
        if 'media_type=video' in line:
            boundary = float(line.split('|')[4].split('=')[-1])
            score = float(line.split('|')[-1].split('=')[-1])
            boundaries.append((boundary, score))
    return boundaries
	
def save_shots(vid_dir_path, shots_dir):
    vid_list = os.listdir(vid_dir_path)
    # count = 0
    for vid in vid_list:
        vid_name = vid.replace('.' + vid.split('.')[-1], '')
        os.mkdir(os.path.join(shots_dir, vid_name))
        b = extract_shots_with_ffprobe(os.path.join(vid_dir_path, vid))
        bounds = [0]
        for num, i in enumerate(b):
            bounds.append(i[0]) 
        print(bounds)
        curr = 0
        t0 = 0
        t1 = 0
        for i in bounds:
            t1 = i 
            if t1 -t0 > 25:
                os.system('ffmpeg -i '+ os.path.join(vid_dir_path, vid) + ' -ss ' + str(t0) + ' -t ' + str(t1-t0) + ' -strict -2 ' +os.path.join(shots_dir, vid_name, str(curr)+'.mp4'))    
                t0 = t1
                t1 = 0
                curr += 1
 
s = time.time()
vid_dir_path = sys.argv[1]
shots_dir = sys.argv[2]
save_shots(vid_dir_path, shots_dir)
e = time.time()
print(e-s)
