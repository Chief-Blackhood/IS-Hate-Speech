import youtube_dl
import pandas as pd
import os
import subprocess

df = pd.read_csv('video_data.csv')

down_titles = []
count = 0
for i in df['ids']:
    count += 1
    if count>3:
        break
    down_url = i
    subprocess.call(['youtube-dl --write-auto-sub --sub-lang en --skip-download \'' + down_url + '\''], shell=True)
    with open('error_file.txt','r') as f:
        line = f.read()
        f.close()
    if "Couldn't find automatic captions" not in line:
        subprocess.call(['youtube-dl \'' + down_url + '\''], shell=True)

    
    print(i)

if not os.path.exists('./videos/'):
    os.mkdir('./videos')
if not os.path.exists('./transcripts/'):
    os.mkdir('./transcripts')

subprocess.call(['mv *.mp4 ./videos/'], shell=True)
subprocess.call(['mv *.m4a ./videos/'], shell=True)
subprocess.call(['mv *.mkv ./videos/'], shell=True)
subprocess.call(['mv *.webm ./videos/'], shell=True)
subprocess.call(['mv *.vtt ./transcripts/'], shell=True)