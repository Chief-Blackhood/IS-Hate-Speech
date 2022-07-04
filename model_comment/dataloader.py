import torch
import torch.utils.data as data
import pandas as pd
import numpy as np


class HateSpeechData(data.Dataset):

    def __init__(self, args, phase):

        self.args = args
        if phase == 'train':
            self.comments = self.load_comments(self.args.train_question_file)
        else:
            self.comments = self.load_comments(self.args.test_question_file)
        if args.add_other_comments:
            self.other_comments_data = self.load_metadata(self.args.other_comments_path)
            self.comments = pd.merge(self.comments, self.other_comments_data, how='left', on=['url', 'comment'])
        if args.add_title or args.add_description or args.add_transcription:
            self.metadata = self.load_metadata(args.metadata_path)
            self.metadata = self.metadata.replace(np.nan, '', regex=True)
            if "bert" in self.args.model:
                if self.args.desc_keyphrase_extract:
                    self.metadata['desc'] = self.metadata['key_phrases_desc_bert']
                if self.args.transcript_keyphrase_extract:
                    self.metadata['transcript'] = self.metadata['key_phrases_transcript_bert']
            elif "longformer" in self.args.model:
                if self.args.desc_keyphrase_extract:
                    self.metadata['desc'] = self.metadata['key_phrases_desc_long']
                if self.args.transcript_keyphrase_extract:
                    self.metadata['transcript'] = self.metadata['key_phrases_transcript_long']
            self.comments = pd.merge(self.comments, self.metadata, how='left', on='url')
        self.comments = self.comments.replace(np.nan, '', regex=True)
        self.comments.drop_duplicates(inplace=True)
        self.comments = self.comments.reset_index(drop=True)

    def load_metadata(self, filename):
        df = pd.read_csv(filename)
        return df

    def load_comments(self, filename):
        df = pd.read_csv(filename)
        df['label'] = df['label'].apply(lambda x: x == 'yes')
        return df

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        comment = self.comments['comment'][index]
        title = self.comments['title'][index] if self.args.add_title else ''
        desc = self.comments['desc'][index] if self.args.add_description else ''
        transcript = self.comments['transcript'][index] if self.args.add_transcription else ''
        other_comment = self.comments['key_phrases_other_comments'][index] if self.args.add_other_comments else ''
        label = self.comments['label'][index]
        return comment, title, desc, transcript, other_comment, torch.FloatTensor([label])
