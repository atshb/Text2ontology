import pickle
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data


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


class Word2vec_Embedding():

    def __init__(self):
        self.w2v   = pd.read_pickle('../data/word2vec/word2vec.pkl')
        self.vocab = pd.read_pickle('../data/wordnet/vocabulary.pkl')
        self.vec_size = self.w2v['example'].shape
        self.pad_vec = np.zeros(self.vec_size)
        self.seq_len = max(len(l.split('_')) for l in self.vocab)

    def __call__(self, batch):
        batch = np.stack([self.vectorize_lemma(l) for l in batch])
        batch = torch.from_numpy(batch).float()
        return batch

    def vectorize_lemma(self, lemma):
        lemma = [self.vectorize_word(w) for w in lemma.split('_')]
        lemma += [self.pad_vec] * (self.seq_len - len(lemma))
        return np.stack(lemma)

    def vectorize_word(self, word):
        if word in self.w2v: return self.w2v[word]
        else               : return self.pad_vec


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
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4*h_size,   y_size)
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
    batch_size = 4

    train = Ont_Dataset(pd.read_pickle('../data/wordnet/train.pkl'))
    valid = Ont_Dataset(pd.read_pickle('../data/wordnet/train.pkl'))

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=False)
    valid_loader = data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    embed = Word2vec_Embedding()

    concat = RNN_ONT()

    for a, b, l in train_loader:
        a, b = embed(a), embed(b)

        print(a.size(), b.size(), l.size())

        h = concat(a, b)

        print(h.size())

        break



if __name__ == '__main__': main()
