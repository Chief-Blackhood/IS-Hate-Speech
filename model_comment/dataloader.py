import torch
import torch.utils.data as data
import pandas as pd


class HateSpeechData(data.Dataset):

    def __init__(self, args, phase, key_bert_model):

        self.args = args
        self.kw_model = key_bert_model
        if phase == 'train':
            self.comments = self.load_comments(args.train_question_file)
        else:
            self.comments = self.load_comments(args.test_question_file)

        if args.add_title or args.add_description or args.add_transcription:
            self.metadata = self.load_metadata(args.metadata_path)
            if args.add_description:
                self.metadata['desc'] = self.metadata['desc'].apply(lambda x: self.process_keyphrase_text(self.preprocess(x), args.desc_word_limit, args.desc_keyphrase_extract, args.desc_key_phrase_count))
            if args.add_transcription:
                self.metadata['transcript'] = self.metadata['transcript'].apply(lambda x: self.process_keyphrase_text(self.preprocess(x), args.transcript_word_limit, args.transcript_keyphrase_extract, args.transcript_key_phrase_count))
            self.comments = pd.merge(self.comments, self.metadata, how='left', on='url')

    def preprocess(self, text):
        if text != text:
            return ''
        new_text = []
    
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def process_keyphrase_text(self, text, max_length, keyphrase_extract, key_phrase_count):
        doc = ' '.join(text.split()[:max_length])
        if keyphrase_extract:
            keywords = self.kw_model.extract_keywords(doc, top_n=key_phrase_count, use_mmr=self.args.use_mmr,
                                                    diversity=self.args.diversity, keyphrase_ngram_range=self.args.keyphrase_ngram_range)
            doc = ' '.join([keyword[0] for keyword in keywords])
        return doc

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
        title = self.comments['title'][index] if self.args.add_title else None
        desc = self.comments['desc'][index] if self.args.add_description else None
        transcript = self.comments['transcript'][index] if self.args.add_transcription else None
        label = self.comments['label'][index]
        return comment, title, desc, transcript, torch.FloatTensor([label])
