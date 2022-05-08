import torch.utils.data as data
import pandas as pd

class HateSpeechData(data.Dataset):

    def __init__(self, args, phase):

        self.args = args
        if phase == 'train':
            self.comments = self.load_comments(args.train_question_file)
        else:
            self.comments = self.load_comments(args.test_question_file)

    def load_comments(self, filename):
        df = pd.read_csv(filename)
        return df

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = self.comments['commment'][index]
        label = self.comments['label'][index]
        return comment, label
