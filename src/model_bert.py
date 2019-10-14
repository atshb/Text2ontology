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
'''
class Ont_Dataset(data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


'''
'''
class BERT_Classifier(nn.Module):

    device = 'cpu'

    def __init__(self, y_size=4):
        super(BERT_Classifier, self).__init__()
        #
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').train()

        self.max_seq_len = 27

        # MLP
        h_size = self.max_seq_len * 768

        self.classifier = nn.Sequential(
            nn.Linear(h_size, h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(h_size, y_size),
        )


    def encode(self, batch_a, batch_b):
        batch_a = [self.tokenizer.tokenize('[CLS]' + a.replace('_', ' ') + '[SEP]') for a in batch_a]
        batch_b = [self.tokenizer.tokenize(          b.replace('_', ' ') + '[SEP]') for b in batch_b]

        batch_tokens = [a + b                   for a, b in zip(batch_a, batch_b)]
        batch_ttypes = [[0]*len(a) + [1]*len(b) for a, b in zip(batch_a, batch_b)]

        batch_tokens = [tks + ['[PAD]'] * (self.max_seq_len - len(tks)) for tks in batch_tokens]
        batch_ttypes = [tps + [0]       * (self.max_seq_len - len(tps)) for tps in batch_ttypes]

        batch_tokens = [self.tokenizer.convert_tokens_to_ids(tks)  for tks in batch_tokens]

        batch_tokens = torch.LongTensor(batch_tokens).to(self.device)
        batch_ttypes = torch.LongTensor(batch_ttypes).to(self.device)

        return batch_tokens, batch_ttypes


    def forward(self, x_a, x_b):
        batch_size = len(x_a)

        tokens, token_types = self.encode(x_a, x_b)

        h = self.bert(tokens, token_type_ids=token_types)[0]
        y = self.classifier(h.view(batch_size, -1))

        return y


    def to(self, device):
        self.device = device
        self.bert.to(device)
        return self



def main():
    batch_size = 3

    train = Ont_Dataset(pd.read_pickle('../data/wordnet/train.pkl'))
    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=False)

    model = BERT_Classifier()

    for a, b, l in train_loader:
        y = model(a, b)
        print(y.size())
        # print(a.size())

        break

if __name__ == '__main__': main()
