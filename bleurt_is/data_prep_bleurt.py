import pandas as pd
import demoji
import re

df_test = pd.read_csv('../data/with_aug/test.csv', usecols=['url', 'comment'])
df_train = pd.read_csv('../data/with_aug/train.csv',  usecols=['url', 'comment'])

def process_text(text=''):
    if text != text:
        return ''
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        t = re.sub(r'http\S+', '', t)
        new_text.append(t)
    new_text = " ".join(new_text)
    new_text = demoji.replace_with_desc(new_text, sep=' ')
    new_text = re.sub('\\s+', ' ', new_text)
    return new_text

df = df_train.append(df_test)
df['comment'] = df['comment'].apply(lambda x: process_text(x))
df.to_csv('bleurt_data.csv', index=False)
