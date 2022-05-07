import torch
from torch import nn
from transformers import LongformerModel, LongformerTokenizer
from transformers import BertTokenizer, BertModel


class LFEmbeddingModule():
    def __init__(self, args, device):
        super(LFEmbeddingModule, self).__init__()
        self.lf_model = BertModel.from_pretrained('bert-large-uncased', output_hidden_states=True).to(device)
        self.lf_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.args = args
        self.device = device
        modules = [self.lf_model.module.embeddings, *self.lf_model.module.encoder.layer[:self.args.freeze_lf_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        
    def get_embeddings(self, comments):
        indexed_cs = []
        max_len = self.args.max_len
        for c in comments:
            ind_c = self.lf_tokenizer.encode(c)[:max_len]
            ind_c.extend((max_len - len(ind_c)) * [self.lf_tokenizer.pad_token_id])
            indexed_cs.append(ind_c)
        indexed_cs = torch.tensor(indexed_cs).to(self.device)
        embedding = self.lf_model(indexed_cs)
        return embedding
    
class CommentModel(nn.Module):
    def __init__(self, args):
        super(CommentModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.max_len, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb):
        out = self.fc(text_emb)
        return out
