import torch
from torch import nn
# from transformers import LongformerModel, LongformerTokenizer
from transformers import BertTokenizer, BertModel

from config import *


class LFEmbeddingModule():
    def __init__(self, args, device):
        super(LFEmbeddingModule, self).__init__()
        self.lf_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True).to(device)
        self.lf_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.args = args
        self.device = device
        modules = [self.lf_model.embeddings, *self.lf_model.encoder.layer[:self.args.freeze_lf_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        
    def get_embeddings(self, comments):
        indexed_cs = []
        max_len = self.args.max_len
        indexed_cs = self.lf_tokenizer.batch_encode_plus(comments, max_length=max_len,padding='max_length', truncation=True)
        indexed_cs = torch.tensor(indexed_cs['input_ids']).to(self.device)
        embedding = self.lf_model(indexed_cs)
        return embedding
    
class CommentModel(nn.Module):
    def __init__(self, args):
        super(CommentModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(FC, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb):
        out = self.fc(text_emb)
        return out
