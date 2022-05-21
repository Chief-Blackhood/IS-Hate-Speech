import torch
import torch.utils.data as data
import pandas as pd
from keybert import KeyBERT

class HateSpeechData(data.Dataset):

    def __init__(self, args, phase):

        self.args = args
        self.kw_model = KeyBERT()
        if phase == 'train':
            self.comments = self.load_comments(args.train_question_file)
        else:
            self.comments = self.load_comments(args.test_question_file)
        if args.add_title or args.add_description:
            self.extra_context = self.load_extra(args.extra_data_path)
            if args.keyphrase_extract:
                self.extra_context['desc'] = self.extra_context['desc'].apply(lambda x: self.process_desc(self.preprocess(x)))
            self.comments = pd.merge(self.comments, self.extra_context, how='left', on='url')

    def preprocess(self, text):
        if text != text:
            return ''
        new_text = []
    
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def process_desc(self, text):
        doc = ' '.join(text.split()[:self.args.desc_word_limit])
        keywords = self.kw_model.extract_keywords(doc, top_n=self.args.key_phrase_count, use_mmr=self.args.use_mmr,
                                             diversity=self.args.diversity, keyphrase_ngram_range=self.args.keyphrase_ngram_range)
        processed_desc = ' '.join([keyword[0] for keyword in keywords])
        return processed_desc

    def load_extra(self, filename):
        df = pd.read_csv(filename)
        return df

    def load_comments(self, filename):
        df = pd.read_csv(filename)
        df['label'] = df['label'].apply(lambda x: x == 'yes')
        return df

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, index):
        context = ''
        comment = self.comments['comment'][index]
        context += comment
        if self.args.add_title:
            title = self.comments['title'][index]
            context += ' [SEP] '+title
        if self.args.add_description:
            desc = self.comments['desc'][index]
            context += ' [SEP] '+desc
        label = self.comments['label'][index]
        return context, torch.FloatTensor([label])
