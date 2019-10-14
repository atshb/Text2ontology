'''
Train model for classification of relationship between Compound words

Usage:
    train_bert.py [--num_train=<nt>] [--num_valid=<nv>] [--batch_size=<bs>] [--max_epoch=<me>]
    train_bert.py (-h | --help)

Options:
    -h --help          show this help message and exit.
    --num_train=<nt>   number of training   data  [default: -1].
    --num_valid=<nv>   number of validation data  [default: -1].
    --batch_size=<bs>  size   of   a   mini-batch [default: 32].
    --max_epoch=<me>   maximum   training   epoch [default: 20].
'''

import pickle
import random
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from model_bert import *
from docopt import docopt


# コマンドライン引数の取得（このファイル上部のドキュメントから自動生成）
args = docopt(__doc__)
num_train  = int(args['--num_train'])
num_valid  = int(args['--num_valid'])
batch_size = int(args['--batch_size'])
max_epoch  = int(args['--max_epoch'])

# データの読み込みとデータセットの作成
train = pd.read_pickle('../data/wordnet/train.pkl')[:num_train]
valid = pd.read_pickle('../data/wordnet/valid.pkl')[:num_valid]
train_loader = data.DataLoader(Ont_Dataset(train), batch_size, shuffle=True)
valid_loader = data.DataLoader(Ont_Dataset(valid), batch_size, shuffle=True)

# 学習モデルの初期化
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BERT_Classifier().to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 学習
for epoch in range(max_epoch):

    # Training
    model.train()
    epoch_loss = 0
    with tqdm(train_loader, ncols=100) as pbar:
        for x_a, x_b, t in pbar:
            #
            t = t.to(device)
            # calculate loss
            y = model(x_a, x_b)
            loss = loss_func(y, t)
            epoch_loss += loss.cpu().item()
            # update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (i + 1) % 1000 == 0: print(f'{i + 1:>4} : {loss.cpu().item():>6.3}')
            outputs = OrderedDict(epoch=f'{int(epoch):>2}', loss=f'{loss.cpu().item():.4f}')
            pbar.set_postfix(outputs)

    # Validation
    model.eval()
    epoch_accu = 0
    with torch.no_grad():
        for x_a, x_b, t in valid_loader:
            #
            t = t.to(device)
            # calculate accuracy
            y = model(x_a, x_b)
            _, y = torch.max(y.data, 1)
            epoch_accu += sum(1 for y_i, t_i in zip(y, t) if y_i == t_i)

    # Show Progress
    epoch_loss /= len(train)
    epoch_accu /= len(valid)
    print(f'{epoch:0>2} | loss : {epoch_loss:>7.5f} | accu : {epoch_accu:.2%}')
