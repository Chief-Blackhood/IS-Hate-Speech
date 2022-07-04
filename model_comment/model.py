from operator import index
from numpy import pad
import torch
from torch import nn
from transformers import LongformerModel, LongformerTokenizer
from transformers import BertTokenizer, BertModel


class LFEmbeddingModule(nn.Module):
    def __init__(self, args, device):
        super(LFEmbeddingModule, self).__init__()
        self.args = args
        if 'longformer' in self.args.model:
            self.lf_model = LongformerModel.from_pretrained(self.args.model, output_hidden_states=True).to(device)
            self.lf_tokenizer = LongformerTokenizer.from_pretrained(self.args.model)
        else:
            self.lf_model = BertModel.from_pretrained(self.args.model, output_hidden_states=True).to(device)
            self.lf_tokenizer = BertTokenizer.from_pretrained(self.args.model)

        self.device = device
        modules = [self.lf_model.embeddings, *self.lf_model.encoder.layer[:self.args.freeze_lf_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        
    def get_embeddings(self, comments, titles, descriptions, transcripts, other_comments):
        indexed_cs = []
        max_len_total = self.args.max_len
        max_len_title = self.args.title_token_count
        max_len_desc = self.args.desc_token_count
        max_len_trans = self.args.transcript_token_count
        max_len_other_comments = self.args.other_comments_token_count
        padding = 'max_length' if self.args.pad_metadata else False
        for comment, title, desc, transcript, other_comment in zip(comments, titles, descriptions, transcripts, other_comments):

            enc_c = self.lf_tokenizer.encode_plus(comment, max_length=max_len_total, padding=False, truncation=True)['input_ids']
            if self.args.add_title:
                enc_t = self.lf_tokenizer.encode_plus(title, max_length=max_len_title, padding=padding, truncation=True)['input_ids']
                enc_c.extend(enc_t[1:])
            if self.args.add_description:
                enc_d = self.lf_tokenizer.encode_plus(desc, max_length=max_len_desc, padding=padding, truncation=True)['input_ids']
                enc_c.extend(enc_d[1:])
            if self.args.add_transcription:
                enc_tr = self.lf_tokenizer.encode_plus(transcript, max_length=max_len_trans, padding=padding, truncation=True)['input_ids']
                enc_c.extend(enc_tr[1:])
            if self.args.add_other_comments:
                enc_oc = self.lf_tokenizer.encode_plus(other_comment, max_length=max_len_other_comments, padding=padding, truncation=True)['input_ids']
                enc_c.extend(enc_oc[1:])
            enc_c = enc_c[:max_len_total]
            enc_c.extend((max_len_total - len(enc_c))*[self.lf_tokenizer.pad_token_id])
            indexed_cs.append(enc_c)
        indexed_cs = torch.tensor(indexed_cs).to(self.device)
        embedding = self.lf_model(indexed_cs)
        return embedding
    
class CommentModel(nn.Module):
    def __init__(self, args):
        super(CommentModel, self).__init__()
        self.args = args
        if 'base' in self.args.model:
            self.fc_size = 768
        else:
            self.fc_size = 1024

        self.fc = nn.Sequential(
            nn.Linear(self.fc_size, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb):
        out = self.fc(text_emb)
        return out
