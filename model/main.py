#!/usr/bin/env python
# coding: utf-8
import argparse
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import ast

from dataloader import HateSpeechData
from model import LFEmbeddingModule, VisionModule, CommentModel
from loss import MultiTaskLoss

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default = 'model_comment/', type = str, help='location of all the train and model files located')
    parser.add_argument("--train_question_file", default='data/with_aug_ttv/train.csv', type=str, help='train data')
    parser.add_argument("--validation_question_file", default='data/with_aug_ttv/validation.csv', type=str, help='validation data')
    parser.add_argument("--test_question_file", default='data/with_aug_ttv/test.csv', type=str, help='test data')
    parser.add_argument("--batch_size", default=8, type=int, help='batch size')
    parser.add_argument("--model", default='bert-large-cased', choices=['bert-large-cased', 'bert-base-cased', 'allenai/longformer-base-4096', 'allenai/longformer-large-4096'], type=str, help='which model to try from bert-large, bert-base and longformer')
    parser.add_argument("--lr", default=0.0003, type=float, help='learning rate')
    parser.add_argument("--num_workers", default=4, type=int, help='number of workers')
    parser.add_argument("--max_epochs", default=1, type=int, help='nummber of maximum epochs to run')
    parser.add_argument("--max_len", default=512, type=int, help='max len of input')
    parser.add_argument("--gpu", default='0', type=str, help='GPUs to use')
    parser.add_argument("--freeze_lf_layers", default=23, type=int, help='number of layers to freeze in BERT or LF')
    parser.add_argument("--metadata_path", default='data/extra_data_trans.csv', type=str, help='metadata for a video')
    parser.add_argument("--pad_metadata", default=True, type=ast.literal_eval, help="need to pad metadata")
    parser.add_argument("--add_comment", default=True, type=ast.literal_eval, help="add original comment")
    parser.add_argument("--add_title", default=False, type=ast.literal_eval, help="add title as context")
    parser.add_argument("--title_token_count", default=50, type=int, help="token to consider of title")
    parser.add_argument("--add_description", default=False, type=ast.literal_eval, help="add description as context")
    parser.add_argument("--desc_keyphrase_extract", default=False, type=ast.literal_eval, help="find key phrase in a doc before adding as context")
    parser.add_argument("--desc_token_count", default=100, type=int, help="number of token to consider of description")
    parser.add_argument("--add_transcription", default=False, type=ast.literal_eval, help="add description as context")
    parser.add_argument("--transcript_keyphrase_extract", default=False, type=ast.literal_eval, help="find key phrase in a doc before adding as context")
    parser.add_argument("--transcript_token_count", default=300, type=int, help="number of token to consider of transcript")
    parser.add_argument("--other_comments_path", default='data/extra_data_other_comments.csv', type=str, help='other comments data for a video')
    parser.add_argument("--add_other_comments", default=False, type=ast.literal_eval, help="add description as context")
    parser.add_argument("--other_comments_token_count", default=300, type=int, help="number of token to consider of transcript")
    parser.add_argument("--add_video", default=False, type=ast.literal_eval, help="add video as context")
    parser.add_argument("--video_path", default='data/videos/', type=str, help='directory which contains videos')
    parser.add_argument("--multitask", default=True, type=ast.literal_eval, help="Flag for multitask classificaiton")
    parser.add_argument("--remove_none", default=False, type=ast.literal_eval, help="Flag for removing Non-Hate in multilabel classification")
    
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

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def accuracy(pred, labels):
    return np.sum(pred == labels)/pred.shape[0]

def save_checkpoint(state, args, name, filename='checkpoint.pth.tar', is_best=False):
    # torch.save(state, filename)
    if is_best:
        lf_filename = os.path.join(args.work_dir, 'lf_model_' + str(name) +'.pth.tar')
        comment_filename = os.path.join(args.work_dir, 'comment_model_' + str(name) +'.pth.tar')
        best_filename = lf_filename if 'lf_model' in filename else comment_filename
        torch.save(state, best_filename)
        # shutil.copyfile(filename, best_filename)

def collate_fn(batch):
    comments, titles, descriptions, transcriptions, other_comments, frames, label_binary, label_multilabel = zip(*batch)

    max_frames = max([image.size(1) for image in frames])
    frames = torch.tensor(np.array([F.pad(image, [0, 0, 0, 0, 0, max_frames - image.size(1)]).numpy() for image in frames]))
    labels = torch.tensor(labels).reshape(-1, 1)
    return [comments, titles, descriptions, transcriptions, other_comments, frames, label_binary, label_multilabel]


def get_data_loaders(args, phase):
    shuffle = True if phase == "train" else False
    data = HateSpeechData(args, phase)
    dataloader = DataLoader(data, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return dataloader

def train_one_epoch(train_loader, epoch, phase, device, criterions, optimizer, lf_model, vision_model, comment_model, multitaskloss_instance, args):
    
    lf_model.lf_model.train()
    vision_model.model.eval()
    comment_model.train()
    multitaskloss_instance.train()

    
    losses = AverageMeter()
    acces = AverageMeter()
    for itr, (comment, title, description, transcription, other_comments, frames, label_binary, label_multilabel) in enumerate(train_loader):
        label_binary = label_binary.to(device)
        label_multilabel = label_multilabel.to(device)

        vis_emb = None
        if args.add_video:
            vis_emb = vision_model.get_embeddings(frames)
        
        output = comment_model(lf_model.get_embeddings(comment, title, description, transcription, other_comments)[1], vis_emb)

        loss_multilabel = criterions[0](output[0], label_multilabel)        
        loss_binary = criterions[1](output[1], label_binary)      

        losses = torch.stack(loss_multilabel, loss_binary)
        multitaskloss = multitaskloss_instance(losses)
        
        optimizer.zero_grad()
        multitaskloss.backward()
        optimizer.step()

        output_binary = np.round(output[1].detach().cpu().numpy())
        label_binary = np.round(label_binary.detach().cpu().numpy())
    
        acc = accuracy(output_binary, label_binary)
        acces.update(acc, args.batch_size)
        
        losses.update(multitaskloss.data.item(), args.batch_size)

        if itr % 25 == 0:
            print(phase + ' Epoch-{:<3d} Iter-{:<3d}/{:<3d}\t'
                'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                epoch, itr, len(train_loader), loss=losses, acc=acces))

    return losses.avg, acces.avg
        
def eval_one_epoch(test_loader, epoch, phase, device, criterions, lf_model, vision_model, comment_model, multitaskloss_instance, args):

    lf_model.lf_model.eval()
    vision_model.model.eval()
    comment_model.eval()
    multitaskloss_instance.eval()

    losses = AverageMeter()
    acces = AverageMeter()
    
    preds = []
    labels = []
    with torch.no_grad():
        for itr, (comment, title, description, transcription, other_comments, frames, label_binary, label_multilabel) in enumerate(test_loader):
            label_binary = label_binary.to(device)
            label_multilabel = label_multilabel.to(device)

            vis_emb = None
            if args.add_video:
                vis_emb = vision_model.get_embeddings(frames)

            output = comment_model(lf_model.get_embeddings(comment, title, description, transcription, other_comments)[1], vis_emb)

            loss_multilabel = criterions[0](output[0], label_multilabel)        
            loss_binary = criterions[1](output[1], label_binary)      

            losses = torch.stack(loss_multilabel, loss_binary)
            multitaskloss = multitaskloss_instance(losses)

            output_binary = output[1].detach().cpu().numpy()
            output_multilabel = output[0].detach().cpu().numpy()

            label_binary = np.round(label_binary.detach().cpu().numpy())
            label_multilabel = np.round(label_multilabel.detach().cpu().numpy())
        
            acc = accuracy(np.round(output_binary), label_binary)
            acces.update(acc, args.batch_size)

            losses.update(multitaskloss.data.item(), args.batch_size)

            preds.append(list([output_binary.tolist(), output_multilabel.tolist()]))
            labels.append(list([label_binary.tolist(), label_multilabel.tolist()]))
        

            if itr % 25 == 0:
                print(phase + ' Epoch-{:<3d} Iter-{:<3d}/{:<3d}\t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'accu {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, itr, len(test_loader), loss=losses, acc=acces))     

    return losses.avg, acces.avg, preds, labels


def load_weights(epoch, lf_model, comment_model, args):
    lf_checkpoint = os.path.join(args.work_dir, 'lf_model_' + str(epoch)+'.pth.tar')
    comment_checkpoint = os.path.join(args.work_dir, 'comment_model_' + str(epoch)+'.pth.tar')
    
    lf_model.lf_model.load_state_dict(torch.load(lf_checkpoint)['state_dict'])
    comment_model.load_state_dict(torch.load(comment_checkpoint)['state_dict'])
    return 
    
def main():  
    args = get_params()
    run = wandb.init(project="vision_models", entity='is_project')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda")
    print('number of available devices:', torch.cuda.device_count())

    train_loader = get_data_loaders(args, 'train')
    validation_loader = get_data_loaders(args, 'validation')
    test_loader = get_data_loaders(args, 'test')
    print('obtained dataloaders')

    lf_model = LFEmbeddingModule(args, device)
    vision_model = VisionModule(args, device)
    comment_model = CommentModel(args).to(device)
    multitaskloss_instance = MultiTaskLoss(n_tasks=2, reduction="sum")

    criterions = [nn.BCEWithLogitsLoss().to(device), nn.BCELoss().to(device)]

    config = wandb.config
    config.lr = args.lr
    wandb.watch(lf_model.lf_model)
    wandb.watch(comment_model)

    params = []
    for model in [lf_model.lf_model, comment_model, multitaskloss_instance]:
        params += list(model.parameters())

    optimizer = optim.Adam(params, lr = args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    print('loaded models')

    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    
    best_eval_loss = np.inf
    train_acc = 0
    eval_acc = 0
    train_loss = 0
    eval_loss = 0
    for epoch in range(args.max_epochs):
        train_loss, train_acc = train_one_epoch(train_loader, epoch, 'Train', device, criterions, optimizer, lf_model, vision_model, comment_model, multitaskloss_instance, args)
        eval_loss, eval_acc, _, _ = eval_one_epoch(validation_loader, epoch, 'Eval', device, criterions, lf_model, vision_model, comment_model, multitaskloss_instance, args)
        print('Epoch-{:<3d} Train: loss {:.4f}\taccu {:.4f}\tEval: loss {:.4f}\taccu {:.4f}'
                .format(epoch, train_loss, train_acc, eval_loss, eval_acc))
        wandb.log({'Train Loss': train_loss, 'Train Acc': train_acc, 'Eval Loss': eval_loss, 'Eval Acc': eval_acc})
        scheduler.step(eval_loss)
        is_better = False
        if eval_loss <= best_eval_loss:
            best_eval_loss = eval_loss
            is_better = True
        
        save_checkpoint({ 'epoch': epoch,
            'state_dict': lf_model.lf_model.state_dict(),
            'best_loss': eval_loss,
            'best_acc' : eval_acc,
            'monitor': 'eval_acc',
            'optimizer': optimizer.state_dict()
        }, args, run.name, os.path.join(args.work_dir, 'lf_model_' + '.pth.tar'), is_better)
        save_checkpoint({ 'epoch': epoch ,
            'state_dict': comment_model.state_dict(),
            'best_loss': eval_loss,
            'best_acc' : eval_acc,
            'monitor': 'eval_acc',
            'vpm_optimizer': optimizer.state_dict()
        }, args, run.name, os.path.join(args.work_dir, 'comment_model_' + '.pth.tar'), is_better)
        
    #load_weights('best')
    test_loss, test_acc, test_pred, test_label = eval_one_epoch(test_loader, 0, 'Test', device, criterions, lf_model, vision_model, comment_model, multitaskloss_instance, args)
    print('Test: loss {:.4f}\taccu {:.4f}'.format(test_loss, test_acc))
    np.save(f'{args.work_dir}/test_preds_{run.name}.npy', np.array(test_pred))
    np.save(f'{args.work_dir}/test_labels_{run.name}.npy', np.array(test_label))

if __name__ == "__main__":
    main()