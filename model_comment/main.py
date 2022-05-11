#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import numpy as np
import shutil

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from dataloader import HateSpeechData
from model import LFEmbeddingModule, CommentModel

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default = 'model_comment/', type = str, help='location of all the train and model files located')
    parser.add_argument("--train_question_file", default='data/with_aug/train.csv', type=str, help='train data')
    parser.add_argument("--test_question_file", default='data/with_aug/test.csv', type=str, help='test data')
    parser.add_argument("--batch_size", default=16, type=int, help='batch size')
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--num_workers", default=4, type=int, help='number of workers')
    parser.add_argument("--max_epochs", default=10, type=int, help='nummber of maximum epochs to run')
    parser.add_argument("--max_len", default=512, type=int, help='max len of input')
    parser.add_argument("--gpu", default='0', type=str, help='GPUs to use')
    parser.add_argument("--freeze_lf_layers", default=23, type=int, help='number of layers to freeze in BERT or LF')
    
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

def accuracy(scores, labels):
    pred = torch.round(scores, -1)
    return torch.sum(pred == labels)/pred.shape[0]

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        best_filename = 'best_lf_model.pth.tar' if 'lf_model' in filename else 'best_comment_model.pth.tar' 
        shutil.copyfile(filename, best_filename)
        
def get_data_loaders(args, phase):
    shuffle = True if phase == "train" else False
    data = HateSpeechData(args, phase)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return dataloader

def train_one_epoch(train_loader, epoch, phase):
    
    lf_model.lf_model.train()
    comment_model.train()
    
    losses = AverageMeter()
    acces = AverageMeter()
    threshold = torch.tensor([0.5]).cpu()
    
    for itr, (comment, label) in enumerate(train_loader):
        input = ['[CLS] ' + c + ' [SEP]' for c in comment]
        output = comment_model(lf_model.get_embeddings(input)[1])
        output = output.cpu()
        output = (output>threshold).float()*1
        label = torch.reshape(label, (16, 1))
        print(type(label[0]), label)
        # output = torch.FloatTensor(output)
        # label = torch.FloatTensor(label)
        loss = criterion(output.to(device), label.to(device))        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc = accuracy(output, label)
        losses.update(loss.data.item(), args.batch_size)
        acces.update(acc.item(), args.batch_size)
        
        if itr % 25 == 0:
            print(phase + ' Epoch-{:<3d} Iter-{:<3d}/{:<3d}\t'
                'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, itr, len(train_loader), loss=losses, acc=acces))        
        
    return losses.avg, acces.avg
        
def eval_one_epoch(data_loader, epoch, phase):
    lf_model.lf_model.eval()
    comment_model.eval()

    losses = AverageMeter()
    acces = AverageMeter()
    threshold = torch.tensor([0.5])
    label_mapping = {"yes": 1., "no": 0.}
    
    preds = []
    labels = []
    with torch.no_grad():
        for itr, (comment, label) in enumerate(train_loader):
            input = ['[CLS] ' + c + ' [SEP]' for c in comment]
            output = comment_model(lf_model.get_embeddings(input)[1])
            output = output.to(device)
            output = (output>threshold).float()*1
            label = [label_mapping[l] for l in label]
            output = torch.FloatTensor(output)
            label = torch.FloatTensor(label)
            loss = criterion(output, label)

            acc = accuracy(output, label)
            losses.update(loss.data.item(), args.batch_size)
            acces.update(acc.item(), args.batch_size)

            preds.extend(list(torch.round(output, -1).numpy()))
            labels.extend(list(label.numpy()))
        
            if itr % 25 == 0:
                print(phase + ' Epoch-{:<3d} Iter-{:<3d}/{:<3d} \t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, itr, len(data_loader), loss=losses, acc=acces))
            
    return losses.avg, acces.avg, preds, labels


def load_weights(epoch):
    lf_checkpoint = os.path.join(args.work_dir, 'lf_model_' + str(epoch)+'.pth.tar')
    comment_checkpoint = os.path.join(args.work_dir, 'comment_model_' + str(epoch)+'.pth.tar')
    
    lf_model.lf_model.load_state_dict(torch.load(lf_checkpoint)['state_dict'])
    comment_model.load_state_dict(torch.load(comment_checkpoint)['state_dict'])
    return 
    
    
args = get_params()
run = wandb.init(project='hatespeech', entity='shrey2809')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda")
print('number of available devices:', torch.cuda.device_count())

train_loader = get_data_loaders(args, 'train')
test_loader = get_data_loaders(args, 'test')
print('obtained dataloaders')

lf_model = LFEmbeddingModule(args, device)
comment_model = CommentModel(args).to(device)
criterion = nn.BCELoss().to(device)

config = wandb.config
config.lr = args.lr
wandb.watch(lf_model.lf_model)
wandb.watch(comment_model)

params = []
for model in [lf_model.lf_model, comment_model]:
    params += list(model.parameters())

optimizer = optim.Adam(params, lr = args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
print('loaded models')

if not os.path.exists(args.work_dir):
    os.mkdir(args.work_dir)

best_eval_acc = 0
train_acc = 0
eval_acc = 0
train_loss = 0
eval_loss = 0
for epoch in range(args.max_epochs):
    train_loss, train_acc = train_one_epoch(train_loader, epoch, 'Train')
    eval_loss, eval_acc, pred, label = eval_one_epoch(test_loader, epoch, 'Eval')
    print('Epoch-{:<3d} Train: loss {:.4f}\taccu {:.4f}\tEval: loss {:.4f}\taccu {:.4f}'
            .format(epoch, train_loss, train_acc, eval_loss, eval_acc))
    wandb.log({'Train Loss': train_loss, 'Train Acc': train_acc, 'Eval Loss': eval_loss, 'Eval Acc': eval_acc})
    scheduler.step(eval_loss)
    is_better = False
    if eval_acc >= best_eval_acc:
        best_eval_acc = eval_acc
        is_better = True
    
    save_checkpoint({ 'epoch': epoch,
        'state_dict': lf_model.lf_model.state_dict(),
        'best_loss': eval_loss,
        'best_acc' : eval_acc,
        'monitor': 'eval_acc',
        'optimizer': optimizer.state_dict()
    }, os.path.join(args.work_dir, 'lf_model_' + str(epoch)+'.pth.tar'), is_better)
    save_checkpoint({ 'epoch': epoch ,
        'state_dict': comment_model.state_dict(),
        'best_loss': eval_loss,
        'best_acc' : eval_acc,
        'monitor': 'eval_acc',
        'vpm_optimizer': optimizer.state_dict()
    }, os.path.join(args.work_dir, 'comment_model_' + str(epoch)+'.pth.tar'), is_better)
    
    
#load_weights('best')
test_loss, test_acc, test_pred, test_label = eval_one_epoch(test_loader, 0, 'Test')
print('Test: loss {:.4f}\taccu {:.4f}'.format(test_loss, test_acc))
print(os.path.join(args.work_dir, 'test_preds.npy'))
np.save(os.path.join(args.work_dir, 'test_preds.npy'), np.array(test_pred))
np.save(os.path.join(args.work_dir, 'test_labels.npy'), np.array(test_label))
