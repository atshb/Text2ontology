'''
Train model for classification of relationship between Compound words

Usage:
    train_rnn.py (-h | --help)
    train_rnn.py [--lr=<lr>]
                 [--seq_len=<sl>]
                 [--max_epoch=<me>]
                 [--batch_size=<bs>]
                 [--num_train=<nt>]
                 [--num_valid=<nv>]

Options:
    -h --help          show this help message and exit.
    --lr=<lr>          leaning rate of optimizer. [default: 1e-5]
    --seq_len=<sl>     maximum sequence length.   [default: 30]
    --max_epoch=<me>   maximum training epoch.    [default: 20]
    --batch_size=<bs>  size of mini-batch.        [default: 32]
    --num_train=<nt>   number of training   data. [default: -1]
    --num_valid=<nv>   number of validation data. [default: -1]
'''


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig, BertForSequenceClassification
from docopt import docopt
from pprint import pprint
from tqdm import tqdm
from dataset import *
from models_rnn import *


'''
'''
def train_model(model, loss_func, optimizer, dataloader, device):
    model.train()

    epoch_loss = 0
    epoch_accu = 0
    with tqdm(dataloader, ncols=64) as pbar:
        for (x_a, x_b), t in pbar:
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            t = t.to(device)
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
            epoch_accu += torch.sum(p == t).item()

    num_data = len(dataloader.dataset)
    return epoch_loss / num_data, epoch_accu / num_data


'''
'''
def valid_model(model, loss_func, optimizer, dataloader, device):
    model.eval()

    epoch_loss = 0
    epoch_accu = 0
    with torch.no_grad():
        for (x_a, x_b), t in dataloader:
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            t = t.to(device)
            # calculate loss
            optimizer.zero_grad()
            y = model(x_a, x_b)
            loss = loss_func(y, t)
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

    lr = float(args['--lr'])
    max_epoch  = int(args['--max_epoch'])
    batch_size = int(args['--batch_size'])
    num_train  = int(args['--num_train'])
    num_valid  = int(args['--num_valid'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データの読み込みとデータセットの作成
    embedding = Word2vecEmbedding()

    train_dataset = WordnetDataset(mode='train', num_data=num_train, transform=embedding)
    valid_dataset = WordnetDataset(mode='valid', num_data=num_valid, transform=embedding)
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=True)

    # 学習モデル
    model = TwinRnnClassifier(300).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学習
    for epoch in range(1, max_epoch+1):
        print('='*27 + f' Epoch {epoch:0>2} ' + '='*27)
        loss, accu = train_model(model, loss_func, optimizer, train_loader, device)
        print(f'|  Training    |  loss-avg : {loss:>8.6f}  |  accuracy : {accu:>8.3%}  |')
        loss, accu = valid_model(model, loss_func, optimizer, valid_loader, device)
        print(f'|  Validation  |  loss-avg : {loss:>8.6f}  |  accuracy : {accu:>8.3%}  |')


if __name__ == '__main__': main()
