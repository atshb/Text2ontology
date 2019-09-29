import pickle
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from transformers import BertModel, BertTokenizer, XLNetModel, XLNetTokenizer

'''
Wordnet_Dataset
    ワードネットから抽出したデータセットを制御するためのクラス。
    preprocess_wordnet.pyを実行してからでないと使えないので注意。
'''
class Ont_Dataset(data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


'''
Word2vec_Embedding
    lemma(複合語)をベクトル(seq_len, 300)に変換するクラス。
    seq_lenは最も単語数の多い複合語。足りない場合は0ベクトルでパディング。
'''
class Word2vec_Embedding():
    device = 'cpu'

    def __init__(self):
        self.w2v   = pd.read_pickle('../data/word2vec/word2vec.pkl')
        self.vocab = pd.read_pickle('../data/wordnet/vocabulary.pkl')
        self.vec_size = self.w2v['example'].shape
        self.pad_vec = np.zeros(self.vec_size)
        self.seq_len = max(len(l.split('_')) for l in self.vocab)

    def __call__(self, batch):
        batch = np.stack([self.vectorize_lemma(l) for l in batch])
        batch = torch.from_numpy(batch)
        return batch.float().to(device)

    def vectorize_lemma(self, lemma):
        words = lemma.split('_')
        vecs =  [self.w2v[w] if w in self.w2v else self.pad_vec for w in words]
        vecs += [self.pad_vec] * (self.seq_len - len(words))
        return np.stack(vecs)

    def to(self, device):
        self.device = device


'''
'''
class BERT_Embedding():
    device = 'cpu'

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').eval().to(device)
        # self.vocab = pd.read_pickle('../data/wordnet/vocabulary.pkl')

    def __call__(self, batch):
        batch = [['[CLS]'] + l.split('_') + ['[SEP]'] for l in batch]
        batch = [self.tokenizer.tokenize(' '.join(l)) for l in batch]

        seq_len = max(len(l) for l in batch)

        batch = [self.tokenizer.convert_tokens_to_ids(l) + [0] * (seq_len - len(l)) for l in batch]
        masks = [[1] * len(l)                            + [0] * (seq_len - len(l)) for l in batch]

        batch = torch.tensor(batch).to(self.device)
        masks = torch.tensor(masks).to(self.device)

        feature, _ = self.bert(batch, None, masks)
        return feature

    def to(self, device):
        self.device = device
        self.model.to(device)

'''
'''
class XLNet_Embedding():
    device = 'cpu'

    def __init__(self):
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.model = XLNetModel.from_pretrained('xlnet-base-cased').eval().to(device)
        # self.vocab = pd.read_pickle('../data/wordnet/vocabulary.pkl')

    def __call__(self, batch):
        batch = [['[CLS]'] + l.split('_') + ['[SEP]'] for l in batch]
        batch = [self.tokenizer.tokenize(' '.join(l)) for l in batch]

        seq_len = max(len(l) for l in batch)

        batch = [self.tokenizer.convert_tokens_to_ids(l) + [0] * (seq_len - len(l)) for l in batch]
        masks = [[1] * len(l)                            + [0] * (seq_len - len(l)) for l in batch]

        batch = torch.tensor(batch).to(self.device)
        masks = torch.tensor(masks).to(self.device)

        feature, _ = self.model(batch, None, masks)
        return feature

    def to(self, device):
        self.device = device
        self.model.to(device)


'''
'''
class RNN_ONT(nn.Module):

    def __init__(self, x_size=300, h_size=300, y_size=4, num_cell=2, drop_rate=0.2):
        self.h_size = h_size
        self.num_cell = 2 # 1のときは修正が必要

        super(RNN_ONT, self).__init__()
        # BiLSTM
        self.lstm_a = nn.LSTM(x_size, h_size, num_layers=num_cell, batch_first=True,
                              dropout=drop_rate, bidirectional=True)
        self.lstm_b = nn.LSTM(x_size, h_size, num_layers=num_cell, batch_first=True,
                              dropout=drop_rate, bidirectional=True)
        # MLP
        self.classifier = nn.Sequential(
            # bidirectional かつ 二つの入力なので hidden size は4倍
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(), nn.Dropout(),
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(), nn.Dropout(),
            nn.Linear(4*h_size,   y_size),
        )

    def forward(self, x_a, x_b):
        _, (h_a, _) = self.lstm_a(x_a)
        _, (h_b, _) = self.lstm_b(x_b)
        h = self.concat(h_a, h_b)
        y = self.classifier(h)
        return y

    def concat(self, a, b):
        _, batch_size, _ = a.size()
        if self.num_cell != 1:
            a = a.view(self.num_cell, 2, batch_size, self.h_size)[-1]
            b = b.view(self.num_cell, 2, batch_size, self.h_size)[-1]
        a = torch.cat([e for e in a], dim=1)
        b = torch.cat([e for e in b], dim=1)
        return torch.cat((a, b), dim=1)


def main():
    batch_size = 64

    train = Ont_Dataset(pd.read_pickle('../data/wordnet/train.pkl'))
    # valid = Ont_Dataset(pd.read_pickle('../data/wordnet/train.pkl'))

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=False)
    # valid_loader = data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # embed = Word2vec_Embedding()
    embed = XLNet_Embedding('cuda')

    for a, b, l in train_loader:
        a = embed(a)
        print(a.size())

        break



if __name__ == '__main__': main()
