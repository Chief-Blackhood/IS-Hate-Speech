import requests
from bs4 import BeautifulSoup
import time

def get_bitchute_data(URL):
    time.sleep(0.01)
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, "html.parser")
    video_data = {"url": URL}

    # Title
    video_data["title"] = soup.find("h1", attrs={"class": "page-title"}).text

    # Numbers and Stats
    headers = {
        "authority": "www.bitchute.com",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": "https://www.bitchute.com",
        "referer": f"{URL}",
        "cookie": "csrftoken=rINo5VQtVi97l3XNI8dMIa9O6rhl0n9TYVdgpIuoievjMfdE0YQ1lnrA7q57z9ng",
        "accept": "*/*",
    }

    data = "csrfmiddlewaretoken=rINo5VQtVi97l3XNI8dMIa9O6rhl0n9TYVdgpIuoievjMfdE0YQ1lnrA7q57z9ng"
    time.sleep(0.01)
    res = requests.post(f"{URL}counts/", data=data, headers=headers).json()
    video_data.update(res)

    # Video Description
    description_div = soup.find(
        "div", attrs={"class": "video-detail-text", "id": "video-description"}
    )
    full_desc = description_div.find("div", attrs={"class": "full hidden"})
    desc = []
    for p in full_desc.find_all("p"):
        desc.append(p.text)
    video_data["description"] = " ".join(desc)
    category_table = soup.find("table", attrs={"class": "video-detail-list"})
    video_data["category"] = category_table.find("a").text

    # Tags
    tags_span = soup.find("span", attrs={"class": "tags", "id": "video-hashtags"})
    tags = []
    for tag in tags_span.find_all("a"):
        tags.append(tag.text[1:])
    video_data["tags"] = tags

    # Video URL to download video
    video = soup.find("video", attrs={"id": "player", "class": "player"})
    video_data["video_url"] = video.source["src"]

    # Comments
    headers = {
        "authority": "commentfreely.bitchute.com",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": "https://www.bitchute.com",
    }
    data = "cf_auth=eyJwcm9maWxlX2lkIjogImFub255bW91cyIsICJvd25lcl9pZCI6ICJmQjRBQWYxZnJoc00iLCAiZGlzcGxheV9uYW1lIjogImFub255bW91cyIsICJ0aHJlYWRfaWQiOiAiYmNfSWVoQTliTTNBUWwxIiwgImljb25fdXJsIjogIi9zdGF0aWMvdjEzNS9pbWFnZXMvYmxhbmstcHJvZmlsZS5wbmciLCAiY2ZfaXNfYWRtaW4iOiAiZmFsc2UifQ%3D%3D+8610c55261a3e3454e5cd41528c55e2e1a55f150761e6753a90b1ff4830366db+1641106687&commentCount=0&isNameValuesArrays=true"

    time.sleep(0.01)
    comments = requests.post(
        "https://commentfreely.bitchute.com/api/get_comments/",
        data=data,
        headers=headers,
    ).json()

    video_data["comments"] = comments
    return video_data
