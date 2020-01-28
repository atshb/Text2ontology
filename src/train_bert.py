'''
Train model for classification of relationship between Compound words

Usage:
    train_bert.py (-h | --help)
    train_bert.py <pretrained_weights>
                  [--dir_name=<dn>]
                  [--lr=<lr>]
                  [--seq_len=<sl>]
                  [--max_epoch=<me>]
                  [--batch_size=<bs>]
                  [--num_train=<nt>]
                  [--num_valid=<nv>]

Options:
    -h --help          show this help message and exit.
    --dir_name=<dn>    Destination directory name. [default: rnn]
    --lr=<lr>          leaning rate of optimizer.  [default: 1e-5]
    --seq_len=<sl>     maximum sequence length.    [default: 50]
    --max_epoch=<me>   maximum training epoch.     [default: 20]
    --batch_size=<bs>  size of mini-batch.         [default: 32]
    --num_train=<nt>   number of training   data.  [default: -1]
    --num_valid=<nv>   number of validation data.  [default: -1]
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import (  BertTokenizer,   BertConfig,   BertForSequenceClassification,
                          AlbertTokenizer, AlbertConfig, AlbertForSequenceClassification)
from docopt import docopt
from pprint import pprint
from tqdm import tqdm
from dataset import *


'''
'''
def train_model(model, optimizer, dataloader, device):
    model.train()

    epoch_loss = 0
    epoch_accu = 0
    with tqdm(dataloader, ncols=64) as pbar:
        for (x, types), t in pbar:
            x = x.to(device)
            t = t.to(device)
            types = types.to(device)
            # calculate loss
            optimizer.zero_grad()
            loss, y = model(x, token_type_ids=types, labels=t)
            _, p = torch.max(y, 1)
            # update model
            loss.backward()
            optimizer.step()
            #
            epoch_loss += loss.cpu().item()
            epoch_accu += torch.sum(p == t).item()

    num_data = len(dataloader.dataset)
    return epoch_loss / num_data, epoch_accu / num_data


'''
'''
def valid_model(model, optimizer, dataloader, device):
    model.eval()

    epoch_loss = 0
    epoch_accu = 0
    with torch.no_grad():
        for (x, types), t in dataloader:
            x = x.to(device)
            t = t.to(device)
            types = types.to(device)
            # calculate loss
            optimizer.zero_grad()
            loss, y = model(x, token_type_ids=types, labels=t)
            _, p = torch.max(y, 1)
            #
            epoch_loss += loss.cpu().item()
            epoch_accu += torch.sum(p == t).item()

    num_data = len(dataloader.dataset)
    return epoch_loss / num_data, epoch_accu / num_data


'''
'''
def main():
    # コマンドライン引数の取得（このファイル上部のドキュメントから自動生成）
    args = docopt(__doc__)
    pprint(args)

    # パラメータの取得
    weights    = args['<pretrained_weights>']
    dir_name   = args['--dir_name']
    lr         = float(args['--lr'])
    seq_len    = int(args['--seq_len'])
    max_epoch  = int(args['--max_epoch'])
    batch_size = int(args['--batch_size'])
    num_train  = int(args['--num_train'])
    num_valid  = int(args['--num_valid'])

    # 学習済みモデルの選択
    weights = args['<pretrained_weights>']

    # モデルアーキテクチャの選択
    if   weights == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(weights)
        config    = BertConfig(num_labels=4)
        model     = BertForSequenceClassification.from_pretrained(weights, config=config)

    elif weights == 'bert-large-uncased':
        tokenizer = BertTokenizer.from_pretrained(weights)
        config    = BertConfig(hidden_size=1024, num_hidden_layers=24,
                               num_attention_heads=16, intermediate_size=4096,
                               num_labels=4)
        model     = BertForSequenceClassification.from_pretrained(weights, config=config)

    elif weights in ['albert-base-v1', 'albert-large-v1', 'albert-base-v2', 'albert-large-v2']:
        tokenizer = AlbertTokenizer.from_pretrained(weights)
        config    = AlbertConfig.from_pretrained(weights, num_labels=4)
        model     = AlbertForSequenceClassification.from_pretrained(weights, config=config)

    # 使用デバイスの取得
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # データの読み込みとデータセットの作成
    encoder = TwinPhraseEncoder(tokenizer, seq_len)

    train_dataset = WordnetDataset(mode='train', num_data=num_train, transform=encoder)
    valid_dataset = WordnetDataset(mode='valid', num_data=num_valid, transform=encoder)
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=True)

    # 最適化法の定義
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 保存用ディレクトリの作成
    log_dir   = f'../logs/{dir_name}'
    model_dir = f'../models/{dir_name}'
    os.makedirs(log_dir  , exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 学習ログの記録（TensorBoard）
    train_writer = SummaryWriter(log_dir=f'{log_dir}/train')
    valid_writer = SummaryWriter(log_dir=f'{log_dir}/valid')

    # 学習
    for epoch in range(1, max_epoch+1):
        print('='*27 + f' Epoch {epoch:0>2} ' + '='*27)

        # Training
        loss, accu = train_model(model, optimizer, train_loader, device)
        print(f'|  Training    |  loss-avg : {loss:>8.6f}  |  accuracy : {accu:>8.3%}  |')
        train_writer.add_scalar('loss', loss, epoch)
        train_writer.add_scalar('accu', accu, epoch)

        # Validation
        loss, accu = valid_model(model, optimizer, valid_loader, device)
        print(f'|  Validation  |  loss-avg : {loss:>8.6f}  |  accuracy : {accu:>8.3%}  |')
        valid_writer.add_scalar('loss', loss, epoch)
        valid_writer.add_scalar('accu', accu, epoch)

        # モデルの保存
        torch.save(model.state_dict(), f'{model_dir}/epoch-{epoch:0>2}.pkl')

    train_writer.close()
    valid_writer.close()


if __name__ == '__main__': main()
