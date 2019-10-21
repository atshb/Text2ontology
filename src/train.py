'''
Train model for classification of relationship between Compound words

Usage:
    train.py (-h | --help)
    train.py (w2v-rnn | bert-ft) [--device=<dv>]
                                 [--max_epoch=<me>]
                                 [--batch_size=<bs>]
                                 [--num_train=<nt>]
                                 [--num_valid=<nv>]

Options:
    -h --help         :  show this help message and exit.
    --device=<dv>     :  device e.g.('cpu', 'cuda', 'cuda:0')
    --max_epoch=<me>  :  maximum training epoch.     [default: 20]
    --batch_size=<bs> :  size of mini-batch.         [default: 32]
    --num_train=<nt>  :  number of training   data.  [default: -1]
    --num_valid=<nv>  :  number of validation data.  [default: -1]
'''

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertForTokenClassification

from docopt import docopt
from pprint import pprint
from tqdm import tqdm

from models import *
from utils  import *


'''
'''
def train_model(model, dataloader, args, loss_func, optimizer):
    model.train()

    epoch_loss = 0
    epoch_right = 0
    epoch_wrong = 0

    with torch.set_grad_enabled(True):
        with tqdm(dataloader, ncols=110) as pbar:
            for x_a, x_b, t in pbar:
                t = t.to(args['--device'])
                # calculate loss
                optimizer.zero_grad()
                y = model(x_a, x_b)
                loss = loss_func(y, t)
                _, p = torch.max(y, 1)
                # update model
                loss.backward()
                optimizer.step()
                #
                epoch_loss += loss.cpu().item()
                epoch_right += torch.sum(p == t.data).item()
                epoch_wrong += torch.sum(p != t.data).item()
                epoch_accu = epoch_right / (epoch_right + epoch_wrong)
                #
                pbar.set_description('Train')
                pbar.set_postfix_str(f'loss:{epoch_loss:>8.2f}, accu:{epoch_accu:>7.2%}')


'''
'''
def valid_model(model, dataloader, args, loss_func, optimizer):
    model.eval()

    epoch_loss = 0
    epoch_right = 0
    epoch_wrong = 0

    with torch.set_grad_enabled(False):
        with tqdm(dataloader, ncols=110) as pbar:
            for x_a, x_b, t in pbar:
                t = t.to(args['--device'])
                # calculate accuracy
                optimizer.zero_grad()
                y = model(x_a, x_b)
                loss = loss_func(y, t)
                _, p = torch.max(y, 1)
                #
                epoch_loss += loss.cpu().item()
                epoch_right += torch.sum(p == t.data).item()
                epoch_wrong += torch.sum(p != t.data).item()
                epoch_accu = epoch_right / (epoch_right + epoch_wrong)
                #
                pbar.set_description('Valid')
                pbar.set_postfix_str(f'loss:{epoch_loss:>8.2f}, accu:{epoch_accu:>7.2%}')


'''
'''
def main():
    # コマンドライン引数の取得（このファイル上部のドキュメントから自動生成）
    args = docopt(__doc__)

    args['--max_epoch' ] = int(args['--max_epoch' ])
    args['--batch_size'] = int(args['--batch_size'])
    args['--num_train' ] = int(args['--num_train' ])
    args['--num_valid' ] = int(args['--num_valid' ])
    #
    if args['--device']: args['--device'] = torch.device(args['--device'])
    else               : args['--device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args['--weights'] = 'bert-base-uncased'
    pprint(args)

    # データの読み込みとデータセットの作成
    train_dataset = Wordnet_dataset(mode='train', num_data=args['--num_train'])
    valid_dataset = Wordnet_dataset(mode='valid', num_data=args['--num_valid'])
    train_loader = data.DataLoader(train_dataset, args['--batch_size'], shuffle=False)
    valid_loader = data.DataLoader(valid_dataset, args['--batch_size'], shuffle=False)

    # 学習モデル
    if   args['w2v-rnn']: model = Word2vecRnnClassifier(args).to(args['--device'])
    elif args['bert-ft']: model = BertClassifier(args).to(args['--device'])

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習
    for epoch in range(args['--max_epoch']):
        print('='*50 + f' Epoch {epoch:0>2} ' + '='*50)
        train_model(model, train_loader, args, loss_func, optimizer)
        valid_model(model, valid_loader, args, loss_func, optimizer)


if __name__ == '__main__': main()
