import os
from pytube import YouTube
    
def get_new_urls(urls):
    try:
        youtube_collected_ids = []
        for file_name in os.listdir("non_hate_youtube_videos"):
            if ".mp4" in file_name:
                youtube_collected_ids.append(file_name.split('.mp4')[0])
    except:
        os.system('mkdir -p non_hate_youtube_videos')
        youtube_collected_ids = []

    youtube_urls = [
        url for url in urls if ("youtube" in url and url.split("?v=")[1].split("&")[0] not in youtube_collected_ids)
    ]
    return youtube_urls

def download_youtube_videos(youtube_urls):
    error_urls = []
    for video_url in youtube_urls:
        try:
            stream = YouTube(video_url).streams.filter(progressive=True, subtype='mp4').order_by('resolution').desc().first()
            if not stream:
                raise
            stream.download(filename=f'non_hate_youtube_videos/{video_url.split("?v=")[1].split("&")[0]}.mp4')
        except:        
            error_urls.append(video_url)
    if len(error_urls):
        try:
            with open('non_hate_youtube_error_urls.txt', 'r') as f:
                original_error_urls = f.read().split('\n')
        except:
            os.system('touch non_hate_youtube_error_urls.txt')
            original_error_urls = []
        error_urls.extend(original_error_urls)
        error_urls = list(set(error_urls))
        with open('non_hate_youtube_error_urls.txt', 'w') as f:
            f.write('\n'.join(error_urls))

with open('url.txt') as file:
    urls = file.readlines()
    urls = [url.split('\n')[0].strip() for url in urls]
    urls = [f"https://www.youtube.com/watch?v={url}" for url in urls]

yt_urls = get_new_urls(urls)
download_youtube_videos(yt_urls)


