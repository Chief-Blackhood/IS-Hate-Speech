import requests
from bs4 import BeautifulSoup
import time
import re

def get_bitchute_data(URL):
    time.sleep(0.01)
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, "html.parser")
    video_data = {"url": URL}
    scripts = soup.findAll('script')
    regex_match = None
    for script in scripts:
        script_tag_contents = script.string
        if script_tag_contents is not None:
            ans = re.search(r"data: {cf_auth: '[^']*", script_tag_contents)
            if ans:
                regex_match = ans.group(0)
    print(URL)
    if regex_match is None:
        print('regex_match is none')
        video_data["comments"] = False
        return video_data
    print(regex_match)
    regex_match = regex_match.split("data: {cf_auth: '")[1]
    try:
        cf_init, cf_extra = regex_match.split('== ')
        cf_d, cf_plus = cf_extra.split(' ')
        cf_auth = cf_init + "%3D%3D+" + cf_d + "+" + cf_plus
    except:
        cf_init, cf_d, cf_plus = regex_match.split(' ')
        cf_auth = cf_init + "%3D%3D+" + cf_d + "+" + cf_plus
    print('cf_auth: '+cf_auth)

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
    
    data = f"cf_auth={cf_auth}&commentCount=0&isNameValuesArrays=true"

    time.sleep(0.01)
    comments = requests.post(
        "https://commentfreely.bitchute.com/api/get_comments/",
        data=data,
        headers=headers,
    ).json()

    video_data["comments"] = comments
    return video_data

# print(get_bitchute_data('https://www.bitchute.com/video/Bb7rY7Id73Dd/'))