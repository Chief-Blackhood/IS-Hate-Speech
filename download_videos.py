import pandas as pd
import urllib.request
import os
from bs4 import BeautifulSoup
import requests
from pytube import YouTube

def load_urls():
    sheet_url = "https://docs.google.com/spreadsheets/d/16lxEwKVA_d_g5QRFNcBTyLz_OBPPB3wZdzZu2UnvLWQ/edit#gid=0"
    url_1 = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")

    data = pd.read_csv(url_1)
    urls = list(data["Link"].dropna())
    return urls

def get_new_urls(urls):
    try:
        bitchute_collected_ids = []
        for file_name in os.listdir("bitchute_videos"):
            if ".mp4" in file_name:
                bitchute_collected_ids.append(file_name.split('.mp4')[0])
    except:
        os.system('mkdir -p bitchute_videos')
        bitchute_collected_ids = []
    try:
        youtube_collected_ids = []
        for file_name in os.listdir("youtube_videos"):
            if ".mp4" in file_name:
                youtube_collected_ids.append(file_name.split('.mp4')[0])
    except:
        os.system('mkdir -p youtube_videos')
        youtube_collected_ids = []

    bitchute_urls = [
    url for url in urls if ("bitchute" in url and url.split('/')[-2] not in bitchute_collected_ids)
    ]
    youtube_urls = [
        url for url in urls if ("youtube" in url and url.split("?v=")[1].split("&")[0] not in youtube_collected_ids)
    ]
    return bitchute_urls, youtube_urls

def download_bitchute_videos(bitchute_urls):
    error_urls = []
    for url in bitchute_urls:
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        video = soup.find("video", attrs={"id": "player", "class": "player"})
        video_src = video.source["src"]
        try:
            urllib.request.urlretrieve(video_src, f'bitchute_videos/{url.split("/")[-2]}.mp4')
        except:
            error_urls.append(url)
    if len(error_urls):
        try:    
            with open('bitchute_error_urls.txt', 'r') as f:
                original_error_urls = f.read().split('\n')
        except:
            os.system('touch bitchute_error_urls.txt')
            original_error_urls = []
        error_urls.extend(original_error_urls)
        error_urls = list(set(error_urls))
        with open('bitchute_error_urls.txt', 'w') as f:
            f.write('\n'.join(error_urls))

def download_youtube_videos(youtube_urls):
    error_urls = []
    for video_url in youtube_urls:
        try:
            stream = YouTube(video_url).streams.filter(progressive=True, subtype='mp4').order_by('resolution').desc().first()
            if not stream:
                raise
            stream.download(filename=f'youtube_videos/{video_url.split("?v=")[1].split("&")[0]}.mp4')
        except:        
            error_urls.append(video_url)
    if len(error_urls):
        try:
            with open('youtube_error_urls.txt', 'r') as f:
                original_error_urls = f.read().split('\n')
        except:
            os.system('touch youtube_error_urls.txt')
            original_error_urls = []
        error_urls.extend(original_error_urls)
        error_urls = list(set(error_urls))
        with open('youtube_error_urls.txt', 'w') as f:
            f.write('\n'.join(error_urls))

urls = load_urls()
bitchute_urls, youtube_urls = get_new_urls(urls)
download_bitchute_videos(bitchute_urls)
download_youtube_videos(youtube_urls)