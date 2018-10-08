import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torchtext.data import Field, BucketIterator, TabularDataset
from selfattention import SelfAttentionGRU

def preprocessing(df, save_path, split_rt=0.0, valid_path=None):
    df.comment_text = df.comment_text.replace('\n', '', regex=True)
    df['targets'] = df[df.iloc[:, 2:].columns].astype(str).apply(lambda x: ','.join(x), axis=1)
    df = df[['comment_text', 'targets']]
    if split_rt > 0.0:
        split_idx = int(len(df) * split_rt)
        df = df.sample(frac=1)
        train = df.iloc[:split_idx, :]
        valid = df.iloc[split_idx:, :]
        train.to_csv(save_path, index=False, header=False, sep='\t')
        valid.to_csv(valid_path, index=False, header=False, sep='\t')
        print('Preprocess complete!')
    else:
        df.to_csv(save_path, index=False, header=False, sep='\t')
    return df


def import_data(config):
    COMMENT = Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = Field(sequential=False, use_vocab=False, preprocessing=lambda x: [int(t) for t in x.split(',')])

    train, test = TabularDataset.splits(path=config.PATH, 
                                        train='train_data.tsv', test='valid_data.tsv',
                                        fields=[('cmt', COMMENT), ('lbl', LABEL)], format='tsv')
    
    COMMENT.build_vocab(train.cmt, min_freq=config.MIN_FREQ)
    print('Data imported ...')
    
    train_iter, test_iter = BucketIterator.splits((train, test), batch_size=config.BATCH, 
                                              device=config.DEVICE, sort_key=lambda x: len(x.cmt), 
                                              sort_within_batch=True, repeat=False)
    print('Iteration build ...')
    return train, test, COMMENT, LABEL, train_iter, test_iter


def build_iteration(config, train, test):
    train_iter, test_iter = BucketIterator.splits((train, test), batch_size=config.BATCH, 
                                              device=config.DEVICE, sort_key=lambda x: len(x.cmt), 
                                              sort_within_batch=True, repeat=False)
    print('Iteration build ...')
    return train_iter, test_iter


def build_model(config, comment_field):
    model = SelfAttentionGRU(vocab_size=len(comment_field.vocab), 
                         embedding_size=config.EMBED, 
                         hidden_size=config.HIDDEN, 
                         fc_hidden=config.FC_HIDDEN, 
                         fc_output=config.FC_OUTPUT, 
                         da=config.DA, 
                         r=config.R, 
                         num_layers=config.NUM_LAYERS, 
                         bidirec=config.BIDRECT).to(config.DEVICE)
    loss_function = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.LAMBDA)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[config.SCHSTEP], optimizer=optimizer)
    print('Model build ...')
    return model, loss_function, optimizer, scheduler


def run_step(model, loader, loss_function, optimizer, device=None):
    model.train()
    acc = 0
    losses=[]
    for batch in loader:
        inputs, lengths = batch.cmt
        targets = batch.lbl        
        model.zero_grad()
        
        scores, loss_P = model(inputs, lengths, device=device)
        
        loss = loss_function(scores, targets.float()) + loss_P
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
    return np.mean(losses)


def validation(model, loader, loss_function, device=None, thres=0.5):
    model.eval()
    acc = 0
    totals = 0
    losses = []    
    for batch in loader:
        inputs, lengths = batch.cmt
        targets = batch.lbl        
        scores, loss_P = model(inputs, lengths, device=device)
        loss = loss_function(scores, targets.float()) + loss_P
        losses.append(loss.item())
        
        preds = torch.sigmoid(scores).ge(thres).long()
        corrects = torch.sum((preds == targets).sum(1).eq(targets.size(1))).item()
        acc += corrects
        totals += targets.size(0)
        
    return np.mean(losses), acc/totals


def train_model(config, model, train_iter, test_iter, loss_function, optimizer, scheduler):
    print('Start Training ...')
    print('--' * 20 )
    start = time.time()
    valid_accs = [0.0]
    for step in range(config.STEP):
        saved = 'False'
        scheduler.step()
        train_loss = run_step(model, train_iter, loss_function, optimizer, device=config.DEVICE)
        valid_loss, valid_acc = validation(model, test_iter, loss_function, device=config.DEVICE)
        valid_accs.append(valid_acc)
        if valid_acc >= max(valid_accs):
            saved = 'True'
            torch.save(model.state_dict(), config.SAVE_PATH)
        
        string = '[{}/{}] train loss: {:.4f}, valid loss: {:.4f}, valid acc: {:.4f} saved: {}'.format(step+1, config.STEP, train_loss, valid_loss, valid_acc, saved)

        print(string)
    end = time.time()
    print('Training time: {:.1f} s'.format(end - start))