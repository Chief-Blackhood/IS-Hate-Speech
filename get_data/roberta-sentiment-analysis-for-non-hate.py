from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np
from scipy.special import softmax
import json
import csv
import urllib.request
from tqdm import tqdm
import pandas as pd

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# print(tf.config.list_physical_devices('GPU'))

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

youtube_comments = {}

with open("non_hate_youtube_data.json", "r") as f:
    youtube_meta_data = json.load(f)

df = pd.read_csv('non-hate-comments-with-emoji.csv', sep='\t')

            
            
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL, from_pt=True)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)
count_comment = 0
count_vid = 0

for idx, data in tqdm(enumerate(youtube_meta_data)):
    
    for in_idex, comment in enumerate(data['comments']):
        text = df.iloc[count_comment]['comment']
        text = preprocess(text)
        # text = text.encode('utf-16', 'surrogatepass').decode('utf-16')
        encoded_input = tokenizer(text, return_tensors='tf')
        output = model(encoded_input)
        scores = output[0][0].numpy()
        scores = softmax(scores)
        youtube_meta_data[idx]['comments'][in_idex]['scores'] = scores.tolist()
        count_comment+=1
    count_vid += 1
    if count_vid % 10 == 0:
       with open("non_hate_youtube_data_with_roberta_scores.json", "w") as f:
            json.dump(youtube_meta_data, f, indent=1) 

with open("non_hate_youtube_data_with_roberta_scores.json", "w") as f:
    json.dump(youtube_meta_data, f, indent=1)
