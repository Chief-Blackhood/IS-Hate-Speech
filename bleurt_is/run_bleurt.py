import tensorflow as tf
import pandas as pd
import json

if tf.test.gpu_device_name():
    print("Default GPU Device: {}".format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

from bleurt import score

checkpoint = "/home/shreyg/BLEURT-20-D12"
scorer = score.BleurtScorer(checkpoint)

references = ["This is a test."]
candidates = ["This is the test."]

scores = scorer.score(references=references, candidates=candidates)
assert type(scores) == list and len(scores) == 1
print(scores)

with open('top_other_comments_all_data.json', 'r') as f:
    reference_comments = json.load(f)

def fetch_bleurt_score(txt, yt_id):
    references = reference_comments[yt_id]
    candidates = [txt] * len(references)
    scores = scorer.score(references=references, candidates=candidates)
    return list(zip(references, scores))

OUTPUT_FILE_NAME = 'bleurt_results.json'

df = pd.read_csv('bleurt_data.csv')

data = []
for comment_ind, comment in df.iterrows():

    print("Post num is ", comment_ind, flush=True)
    if comment_ind % 100 == 3:
        with open(OUTPUT_FILE_NAME, "w") as fd:
            json.dump(data, fd)
            print("Saving", flush=True)

    data.append({
        "url": comment['url'],
        "comment": comment['comment'],
        "scores": fetch_bleurt_score(comment['comment'], comment['url'])
    })

with open(OUTPUT_FILE_NAME, "w") as fd:
    json.dump(data, fd, indent=2)
    print("FInal done", flush=True)