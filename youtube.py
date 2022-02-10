import requests

my_api_key = "AIzaSyCyegJgVEgui6-DZ56Ybg3trtC0UUjJvKQ"


def fetch_comments_with_replies(video_id):
 
    MAX_RESULTS=100

    base_url=f'https://youtube.googleapis.com/youtube/v3/commentThreads?part=id&part=replies&part=snippet&maxResults={MAX_RESULTS}&order=time&videoId={video_id}&key={my_api_key}'
    final_ans=[]
    next_page_token=None
    
    while True:
        final_url=base_url
        if next_page_token is not None:
            final_url+=f"&pageToken={next_page_token}"
        _res = requests.get(final_url, headers={"Accept": "application/json"})
        json = _res.json()
        final_json = []
        if 'items' in json:
            for raw_comment in json['items']:
                top_comment = raw_comment['snippet']['topLevelComment']['snippet']
                top_comment['id'] = raw_comment['snippet']['topLevelComment']['id']
                top_comment['totalReplyCount'] = raw_comment['snippet']['totalReplyCount']

                to_remove = ['videoId', 'authorChannelId', 'authorChannelUrl', 'authorProfileImageUrl', 'canRate', 'viewerRating', 'publishedAt', 'updatedAt', 'textDisplay']
                for feature in to_remove:
                    top_comment.pop(feature, None)
                
                comment = {"top_comment": top_comment}
                if 'replies' not in raw_comment:
                    continue
                if len(raw_comment['replies']['comments']) == top_comment['totalReplyCount']:
                    raw_replies = raw_comment['replies']['comments']
                else:
                    comment_list_url = f'https://youtube.googleapis.com/youtube/v3/comments?part=snippet&part=id&maxResults=100&parentId={top_comment["id"]}&key={my_api_key}'
                    replies_res = requests.get(comment_list_url, headers={"Accept": "application/json"})
                    raw_replies = replies_res.json()['items']
                replies = []
                for reply in raw_replies:
                    rep = reply['snippet']
                    rep['id'] = reply['id']
                    for feature in to_remove:
                        rep.pop(feature, None)
                    replies.append(rep)
                comment['replies'] = replies
                final_json.append(comment)
        final_ans.extend(final_json)
        if "nextPageToken"  in json.keys():
            next_page_token=json["nextPageToken"]
        else:
            break
    return final_ans


def fetch_video_details(video_id):
    base_url = f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2C%20statistics&id={video_id}&key={my_api_key}"
    final_data = {}
    _res = requests.get(base_url, headers={"Accept": "application/json"})
    json = _res.json()
    if "items" not in json:
        return final_data
    api_data = json["items"][0]
    to_extract = {
        "snippet": ["publishedAt", "title", "description", "tags", "categoryId"],
        "statistics": ["viewCount", "likeCount", "commentCount"],
    }
    for key, types in to_extract.items():
        for type in types:
            final_data[type] = api_data[key][type]
    return final_data


def get_youtube_data(URL):

    video_data = {"url": URL}
    vid_id = URL.split("?v=")[1].split("&")[0]

    comments = fetch_comments_with_replies(vid_id)
    stats = fetch_video_details(vid_id)

    if stats:
        video_data["stats"] = stats

    if len(comments) != 0:
        video_data["comments"] = comments

    return video_data