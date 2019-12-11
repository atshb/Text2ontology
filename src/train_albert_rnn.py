'''
Train model for classification of relationship between Compound words

Usage:
    train_bert_rnn.py (-h | --help)
    train_bert_rnn.py (rnn | cnn | series | parallel)
                      [--lr=<lr>]
                      [--seq_len=<sl>]
                      [--max_epoch=<me>]
                      [--batch_size=<bs>]
                      [--num_train=<nt>]
                      [--num_valid=<nv>]

Options:
    -h --help          show this help message and exit.
    --lr=<lr>          leaning rate of optimizer. [default: 1e-3]
    --seq_len=<sl>     maximum sequence length.   [default: 30]
    --max_epoch=<me>   maximum training epoch.    [default: 20]
    --batch_size=<bs>  size of mini-batch.        [default: 64]
    --num_train=<nt>   number of training   data. [default: -1]
    --num_valid=<nv>   number of validation data. [default: -1]
'''

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AlbertTokenizer, AlbertModel

from docopt import docopt
from pprint import pprint
from tqdm import tqdm
from dataset import *
from models_rnn import *

'''
'''
def train_model(model, embed, loss_func, optimizer, dataloader, device):
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
            with torch.no_grad():
                e_a = embed(x_a)[0]
                e_b = embed(x_b)[0]
            y = model(e_a, e_b)
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
def valid_model(model, embed, loss_func, optimizer, dataloader, device):
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
            with torch.no_grad():
                e_a = embed(x_a)[0]
                e_b = embed(x_b)[0]
            y = model(e_a, e_b)
            loss = loss_func(y, t)
            _, p = torch.max(y, 1)

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
    lr = float(args['--lr'])
    seq_len    = int(args['--seq_len'])
    max_epoch  = int(args['--max_epoch'])
    batch_size = int(args['--batch_size'])
    num_train  = int(args['--num_train'])
    num_valid  = int(args['--num_valid'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルの選択
    vec_size = 768
    pretrained_weights = 'albert-base-v1'
    tokenizer = AlbertTokenizer.from_pretrained(pretrained_weights)
    albert_emb = AlbertModel.from_pretrained(pretrained_weights)
    albert_emb.to(device).eval()

    # 学習モデル
    if   args['rnn']    : model = RnnClassifier(vec_size)
    elif args['cnn']    : model = CnnClassifier(vec_size)
    elif args['series']  : model = SeriesClassifer(seq_len, vec_size)
    elif args['parallel']: model = ParallelClassifier(vec_size)
    model.to(device)

    # データの読み込みとデータセットの作成
    encoder = SeparatePhraseEncoder(tokenizer, seq_len)

    train_dataset = WordnetDataset(mode='train', num_data=num_train, transform=encoder)
    valid_dataset = WordnetDataset(mode='valid', num_data=num_valid, transform=encoder)
    train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=True)

    # 最適化法の定義
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学習
    for epoch in range(1, max_epoch+1):
        print('='*27 + f' Epoch {epoch:0>2} ' + '='*27)
        # Training
        loss, accu = train_model(model, albert_emb, loss_func, optimizer, train_loader, device)
        print(f'|  Training    |  loss-avg : {loss:>8.6f}  |  accuracy : {accu:>8.3%}  |')
        # Validation
        loss, accu = valid_model(model, albert_emb, loss_func, optimizer, valid_loader, device)
        print(f'|  Validation  |  loss-avg : {loss:>8.6f}  |  accuracy : {accu:>8.3%}  |')
        # 保存
        torch.save(model.state_dict(), f'../result/bert.pkl')


if __name__ == '__main__': main()
