'''
Train model for classification of relationship between Compound words

Usage:
    train_bert.py [--num_train=<nt>] [--num_valid=<nv>] [--batch_size=<bs>] [--max_epoch=<me>]
    train_bert.py (-h | --help)

Options:
    -h --help          show this help message and exit.
    --num_train=<nt>   number of training   data  [default: -1].
    --num_valid=<nv>   number of validation data  [default: -1].
    --batch_size=<bs>  size    of    mini-batch   [default: 32].
    --max_epoch=<me>   maximum  training  epoch   [default: 20].
'''

import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertForTokenClassification

from tqdm import tqdm
from docopt import docopt

from models import *
from utils  import *


'''
'''
def train_model(model, dataloader, device, loss_func, optimizer):
    model.train()
    epoch_loss = 0

    with torch.set_grad_enabled(True):
        with tqdm(dataloader, ncols=100) as pbar:
            for x_a, x_b, t in pbar:
                t = t.to(device)
                # calculate loss
                y = model(x_a, x_b)
                loss = loss_func(y, t)
                epoch_loss += loss.cpu().item()
                # update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    epoch_loss /= len(dataloader.dataset)
    return epoch_loss


'''
'''
def valid_model(model, dataloader, device):
    model.eval()
    epoch_accu = 0

    with torch.set_grad_enabled(False):
        with tqdm(dataloader, ncols=100) as pbar:
            for x_a, x_b, t in pbar:
                t = t.to(device)
                # calculate accuracy
                y = model(x_a, x_b)
                _, y = torch.max(y.data, 1)
                epoch_accu += sum(1 for y_i, t_i in zip(y, t) if y_i == t_i)

    epoch_accu /= len(dataloader.dataset)
    return epoch_accu


'''
'''
def main():
    # コマンドライン引数の取得（このファイル上部のドキュメントから自動生成）
    args = docopt(__doc__)
    num_train  = int(args['--num_train'])
    num_valid  = int(args['--num_valid'])
    batch_size = int(args['--batch_size'])
    max_epoch  = int(args['--max_epoch'])

    # データの読み込みとデータセットの作成
    train_dataset = Wordnet_dataset(mode='train', num_data=num_train)
    valid_dataset = Wordnet_dataset(mode='valid', num_data=num_valid)
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=True)

    # 学習モデルの初期化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained_weights = 'bert-base-uncased'
    model = BertClassifier(pretrained_weights).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 学習
    for epoch in range(max_epoch):
        loss = train_model(model, train_loader, device, loss_func, optimizer)
        print(f'train-{epoch:0>2} | loss : {loss:>6.4f}')

        accu = valid_model(model, valid_loader, device)
        print(f'valid-{epoch:0>2} | accu : {accu:.2%}')


if __name__ == '__main__': main()
