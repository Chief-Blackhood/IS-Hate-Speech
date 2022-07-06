import pandas as pd
import re

def load_sheet(url):
    url_1 = url.replace("/edit#gid=", "/export?format=csv&gid=")
    data = pd.read_csv(url_1)
    return data


HATE_SHEET = "https://docs.google.com/spreadsheets/d/16lxEwKVA_d_g5QRFNcBTyLz_OBPPB3wZdzZu2UnvLWQ/edit#gid=0"
POS_NON_HATE_SHEET = "https://docs.google.com/spreadsheets/d/16lxEwKVA_d_g5QRFNcBTyLz_OBPPB3wZdzZu2UnvLWQ/edit#gid=1070451623"
NEU_NON_HATE_SHEET = "https://docs.google.com/spreadsheets/d/16lxEwKVA_d_g5QRFNcBTyLz_OBPPB3wZdzZu2UnvLWQ/edit#gid=497253390"

hate_df = load_sheet(HATE_SHEET)
pos_df = load_sheet(POS_NON_HATE_SHEET)
neu_df = load_sheet(NEU_NON_HATE_SHEET)
non_hate_df = pd.concat([pos_df, neu_df])

hate_df.drop(
    columns=[
        "Title",
        "Is Video Hateful (Yes / No)",
        "What Metadata / Information is Required?",
        "Synthetic or Original?",
        "Reviewer",
        "Additional Verification Needed (Yes / No)",
        "Reason For Additional Verficiation? (Only if YES)",
        "Video Category"
    ],
    inplace=True,
)
hate_df.rename(
    columns={
        "Link": "url",
        "Comment": "comment",
        "Hate Towards Whom?": "target"
    },
    inplace=True,
)
hate_df.fillna(method="ffill", inplace=True)

non_hate_df.drop(
    columns=[
        "Manual Inspection",
        "Validator",
        "scores",
        "type"
    ],
    inplace=True,
)

def assign(text):
    if text != text:
        return 'None'
    else:
        return text

df = pd.concat([hate_df, non_hate_df])
df['comment'] = df['comment'].apply(lambda x: " ".join(x.split()))
train_df = pd.read_csv('data/without_aug/train.csv')
test_df = pd.read_csv('data/without_aug/test.csv')
train_df['comment'] = train_df['comment'].apply(lambda x: " ".join(x.split()))
train_df = train_df[train_df["comment"] != '']
test_df['comment'] = test_df['comment'].apply(lambda x: " ".join(x.split()))
train_df = pd.merge(train_df, df, how='left', on=['url', 'comment'])
test_df = pd.merge(test_df, df, how='left', on=['url', 'comment'])
test_df['target'] = test_df['target'].apply(lambda x: assign(x))
train_df.to_csv('data/hate_towards_whom_org/train.csv', index=False)
test_df.to_csv('data/hate_towards_whom_org/test.csv', index=False)

train_df = pd.read_csv('data/hate_towards_whom_org/train.csv')

def check(text):
    if text in ['Individual', 'Organisation', 'Location', 'Community', 'None']:
        return True
    return False

train_aug_df = pd.read_csv('data/hate_towards_whom_aug/old_train.csv')
train_aug_df['comment'] = train_aug_df['comment'].apply(lambda x: " ".join(x.split()))
train_aug_df = train_aug_df[train_aug_df['comment'] != '']
train_aug_df = pd.merge(train_aug_df, train_df, how='left', on=['url', 'category', 'comment', 'label'])
target = []
prev_target = "None"
for index, row in train_aug_df.iterrows():
    if ~row.isna().any() and row['label'] == 'yes' and row['target'] is not None and row['target'] != '':
        prev_target = row['target']
        prev_target = re.sub(r"[\(\[].*?[\)\]]", "", prev_target)
        prev_target = re.sub(r",", "", prev_target)
        prev_target = ",".join([cat for cat in sorted(list(set(prev_target.split()))) if check(cat)])
        target.append(prev_target)
    elif row['label'] == 'yes':
        target.append(prev_target)
    else:
        prev_target = "None"
        target.append(prev_target)

train_aug_df['target'] = target
train_aug_df.to_csv('data/hate_towards_whom_aug/train.csv', index=False)


def clean(text):
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    text = re.sub(r",", "", text)
    text = ",".join([cat for cat in sorted(list(set(text.split()))) if check(cat)])
    return text
        
test_aug_df = pd.read_csv('data/hate_towards_whom_org/test.csv')
test_aug_df['target'] = test_aug_df['target'].apply(lambda x: clean(x))

test_aug_df.to_csv('data/hate_towards_whom_aug/test.csv', index=False)
print(test_aug_df['target'].value_counts())
print(train_aug_df['target'].value_counts())