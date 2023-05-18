import json

import pandas as pd

from bitchute import get_bitchute_data
from youtube import get_youtube_data
from tqdm import tqdm
from datetime import datetime


sheet_url = "https://docs.google.com/spreadsheets/d/16lxEwKVA_d_g5QRFNcBTyLz_OBPPB3wZdzZu2UnvLWQ/edit#gid=0"
url_1 = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")

data = pd.read_csv(url_1)
urls = list(data["Link"].dropna())

# try:
#     with open("bitchute_data.json", "r") as f:
#         bitchute_meta_data = json.load(f)
#         bitchute_collected_urls = [data["url"] for data in bitchute_meta_data]
# except:
bitchute_meta_data = []
bitchute_collected_urls = []

# try:
#     with open("./metadata/youtube_data.json", "r") as f:
#         youtube_meta_data = json.load(f)
#         youtube_collected_urls = [data["url"] for data in youtube_meta_data]
# except:
#     print('fuck up')
#     youtube_meta_data = []
#     youtube_collected_urls = []

bitchute_urls = [
    url for url in urls if ("bitchute" in url)# and url not in bitchute_collected_urls)
]
# youtube_urls = [
#     url for url in urls if ("youtube" in url and url not in youtube_collected_urls)
# ]
print(f"Date and time: {datetime.now()}")

new_data = []
count = 0
# for url in tqdm(youtube_urls[:50]):
#     count+=1
#     data = get_youtube_data(url)
#     if "stats" in data and "comments" in data:
#         new_data.append(data)
#     if count % 10 == 0:
#        youtube_meta_data.extend(new_data)
#        new_data = [] 
#        with open("youtube_data.json", "w") as f:
#            json.dump(youtube_meta_data, f, indent=1)
with open("./metadata/bitchute_data.json", "r") as f:
    old_bitchute_data = json.load(f)

for url in tqdm(bitchute_urls):
    video_data = get_bitchute_data(url)
    if not video_data['comments']:
        for old_video_data in old_bitchute_data:
            if old_video_data['url'] == url:
                old_video_data['comments'] = {}
                bitchute_meta_data.extend([old_video_data])
    else:
        bitchute_meta_data.extend([video_data])
# bitchute_meta_data.extend([get_bitchute_data(url) for url in tqdm(bitchute_urls)])
# youtube_meta_data.extend(new_data)

    with open("bitchute_data.json", "w") as f:
        json.dump(bitchute_meta_data, f, indent=1)

# with open("youtube_data.json", "w") as f:
#     json.dump(youtube_meta_data, f, indent=1)

# https://www.bitchute.com/video/CBfFtJ04hk77/
