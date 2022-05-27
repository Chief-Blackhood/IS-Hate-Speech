import torch
from torch import nn
from transformers import LongformerModel, LongformerTokenizer
from transformers import BertTokenizer, BertModel


class LFEmbeddingModule():
    def __init__(self, args, device):
        super(LFEmbeddingModule, self).__init__()
        self.args = args
        if 'longformer' in self.args.model:
            self.lf_model = LongformerModel.from_pretrained(self.args.model, output_hidden_states=True).to(device)
            self.lf_tokenizer = LongformerTokenizer.from_pretrained(self.args.model)
        else:
            self.lf_model = BertModel.from_pretrained(self.args.model, output_hidden_states=True).to(device)
            self.lf_tokenizer = BertTokenizer.from_pretrained(self.args.model)

        if 'base' in self.args.model:
            self.fc_size = 768
        else:
            self.fc_size = 1024

        self.device = device
        modules = [self.lf_model.embeddings, *self.lf_model.encoder.layer[:self.args.freeze_lf_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        
    def get_embeddings(self, comments, titles, descriptions, transcripts):
        indexed_cs = []
        max_len = self.args.max_len
        padding = 'max_length' if self.args.pad_metadata else False
        for comment, title, desc, transcript in zip(comments, titles, descriptions, transcripts):
            enc_c = self.lf_tokenizer.encode(comment)
            if self.args.add_title:
                enc_t = self.lf_tokenizer.encode(title, max_len=self.args.title_token_count, padding=padding, truncation=True)
                enc_c.extend(enc_t[1:])
            if self.args.add_description:
                enc_d = self.lf_tokenizer.encode(desc, max_len=self.args.desc_token_count, padding=padding, truncation=True)
                enc_c.extend(enc_d[1:])
            if self.args.add_transcript:
                enc_tr = self.lf_tokenizer.encode(transcript, max_len=self.args.transcript_token_count, padding=padding, truncation=True)
                enc_c.extend(enc_tr[1:])

            enc_c.append((max_len - len(enc_c))*[self.lf_tokenizer.pad_token_id])

        indexed_cs = torch.tensor(indexed_cs).to(self.device)
        embedding = self.lf_model(indexed_cs)
        return embedding
    
class CommentModel(nn.Module):
    def __init__(self, args):
        super(CommentModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(self.fc_size, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb):
        out = self.fc(text_emb)
        return out
