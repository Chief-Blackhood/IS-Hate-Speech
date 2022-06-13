import os
import torch
import argparse
import ast
import numpy as np
from model import LFEmbeddingModule, CommentModel
from torch import nn
from dataloader import HateSpeechData
from torch.utils.data import DataLoader

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default = 'model_comment/', type = str, help='location of all the train and model files located')
    parser.add_argument("--train_question_file", default='data/with_aug/train.csv', type=str, help='train data')
    parser.add_argument("--test_question_file", default='data/with_aug/test.csv', type=str, help='test data')
    parser.add_argument("--batch_size", default=8, type=int, help='batch size')
    parser.add_argument("--model", default='bert-large-uncased', choices=['bert-large-uncased', 'bert-base-uncased', 'allenai/longformer-base-4096', 'allenai/longformer-large-4096'], type=str, help='which model to try from bert-large, bert-base and longformer')
    parser.add_argument("--lr", default=0.0003, type=float, help='learning rate')
    parser.add_argument("--num_workers", default=0, type=int, help='number of workers')
    parser.add_argument("--max_epochs", default=1, type=int, help='nummber of maximum epochs to run')
    parser.add_argument("--max_len", default=512, type=int, help='max len of input')
    parser.add_argument("--gpu", default='0', type=str, help='GPUs to use')
    parser.add_argument("--freeze_lf_layers", default=23, type=int, help='number of layers to freeze in BERT or LF')
    parser.add_argument("--metadata_path", default='data/extra_data_trans.csv', type=str, help='metadata for a video')
    parser.add_argument("--pad_metadata", default=True, type=ast.literal_eval, help="need to pad metadata")
    parser.add_argument("--add_title", default=False, type=ast.literal_eval, help="add title as context")
    parser.add_argument("--title_token_count", default=50, type=int, help="token to consider of title")
    parser.add_argument("--add_description", default=False, type=ast.literal_eval, help="add description as context")
    parser.add_argument("--desc_keyphrase_extract", default=False, type=ast.literal_eval, help="find key phrase in a doc before adding as context")
    parser.add_argument("--desc_token_count", default=100, type=int, help="number of token to consider of description")
    parser.add_argument("--add_transcription", default=False, type=ast.literal_eval, help="add description as context")
    parser.add_argument("--transcript_keyphrase_extract", default=False, type=ast.literal_eval, help="find key phrase in a doc before adding as context")
    parser.add_argument("--transcript_token_count", default=300, type=int, help="number of token to consider of transcript")
    
    return parser.parse_args()

class AverageMeter(object):

    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(pred, labels):
    return np.sum(pred == labels)/pred.shape[0]

def get_data_loaders(args, phase):
    shuffle = True if phase == "train" else False
    data = HateSpeechData(args, phase)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return dataloader

def load_weights(lf_model, comment_model):
    lf_checkpoint = os.path.join('./models/lf_model_dry_shape_68.pth.tar')
    comment_checkpoint = os.path.join('./models/comment_model_dry_shape_68.pth.tar')
    
    lf_model.lf_model.load_state_dict(torch.load(lf_checkpoint)['state_dict'])
    comment_model.load_state_dict(torch.load(comment_checkpoint)['state_dict'])
    return lf_model, comment_model

def eval_one_epoch(test_loader, epoch, phase, device, criterion, lf_model, comment_model, args):
    lf_model.lf_model.eval()
    comment_model.eval()

    losses = AverageMeter()
    acces = AverageMeter()
    
    preds = []
    labels = []
    with torch.no_grad():
        for itr, (comment, title, description, transcription, label) in enumerate(test_loader):
            label = label.to(device)

            output = comment_model(lf_model.get_embeddings(comment, title, description, transcription)[1])

            loss = criterion(output, label)

            output = np.round(output.detach().cpu().numpy())
            label = np.round(label.detach().cpu().numpy())
        

            acc = accuracy(output, label)
            losses.update(loss.data.item(), args.batch_size)
            acces.update(acc, args.batch_size)

            preds.extend(list(output))
            labels.extend(list(label))
        
            if itr % 25 == 0:
                print(phase + ' Epoch-{:<3d} Iter-{:<3d}/{:<3d} \t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, itr, len(test_loader), loss=losses, acc=acces))
            
    return losses.avg, acces.avg, preds, labels

args = get_params()
test_loader = get_data_loaders(args, 'test')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lf_model = LFEmbeddingModule(args, device)
comment_model = CommentModel(args).to(device)
criterion = nn.BCELoss().to(device)

test_loss, test_acc, test_pred, test_label = eval_one_epoch(test_loader, 0, 'Test', device, criterion, lf_model, comment_model, args)
print('Test: loss {:.4f}\taccu {:.4f}'.format(test_loss, test_acc))
np.save('./test_preds.npy', np.array(test_pred))