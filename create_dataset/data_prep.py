from operator import index
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from tqdm import tqdm


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
        "Hate Towards Whom?",
    ],
    inplace=True,
)
hate_df.rename(
    columns={
        "Link": "url",
        "Video Category": "category",
        "Comment": "comment",
    },
    inplace=True,
)
hate_df["category"] = hate_df["category"].str.lower()
hate_df["label"] = "yes"
hate_df.fillna(method="ffill", inplace=True)

non_hate_df.drop(
    columns=[
        "Manual Inspection",
        "Validator",
        "scores",
    ],
    inplace=True,
)
non_hate_df.rename(
    columns={
        "type": "category",
    },
    inplace=True,
)
non_hate_df["category"] = non_hate_df["category"].str.lower()
non_hate_df["label"] = "no"

df = pd.concat([hate_df, non_hate_df])
df.drop(columns=['Unnamed: 11'], inplace=True)

# df["strat"] = df["category"] + " " + df["label"]
# y = df["label"]
# X = df.drop(columns=["label"])


groups = df['url']

best_state = 0
min_diff = 1000000
for random_state in tqdm(range(0, 1000)):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)

    for train_index, test_index in gss.split(df, groups=groups):
        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]

    try:
        count_train = df_train['category'].value_counts().to_dict()
        count_test = df_test['category'].value_counts().to_dict()
        diff = {k : count_train[k] - count_test[k] * 9 for k in count_train}
        value_sum = sum(map(abs, diff.values()))
        if value_sum < min_diff:
            best_state = random_state
            min_diff = value_sum
    except:
        continue

print(best_state, min_diff)
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=best_state)

for train_index, test_index in gss.split(df, groups=groups):
    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, shuffle=True, stratify=X["strat"]
# )

# X_train.drop(columns=["strat"], inplace=True)
# X_test.drop(columns=["strat"], inplace=True)
# df_train = pd.concat([X_train, y_train], axis=1)
# df_test = pd.concat([X_test, y_test], axis=1)
df_train.to_csv("data/without_aug/train.csv", index=False)
df_test.to_csv("data/without_aug/test.csv", index=False)
