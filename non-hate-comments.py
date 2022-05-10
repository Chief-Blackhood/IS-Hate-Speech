from tqdm import tqdm
from youtube import get_youtube_data
import json

with open('url.txt') as file:
    urls = file.readlines()
    urls = [url.split('\n')[0].strip() for url in urls]
    urls = [f"https://www.youtube.com/watch?v={url}" for url in urls]

try:
    with open("non_hate_youtube_data.json", "r") as f:
        youtube_meta_data = json.load(f)
        youtube_collected_urls = [data["url"] for data in youtube_meta_data]
except:
    youtube_meta_data = []
    youtube_collected_urls = []
youtube_urls = [
    url for url in urls if ("youtube" in url and url not in youtube_collected_urls)
]

count = 0
for url in tqdm(youtube_urls):
    print(url)
    data = get_youtube_data(url)
    if "stats" in data and "comments" in data:
        youtube_meta_data.append(data)
    count += 1
    if count % 10 == 0:
        with open("non_hate_youtube_data.json", "w") as f:
            json.dump(youtube_meta_data, f, indent=1)

with open("non_hate_youtube_data.json", "w") as f:
    json.dump(youtube_meta_data, f, indent=1)
