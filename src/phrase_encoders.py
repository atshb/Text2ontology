import pickle
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from transformers import BertModel, BertTokenizer, BertForTokenClassification, XLNetModel, XLNetTokenizer



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
class BERT_Classifier2(nn.Module):

    device = 'cpu'
    max_seq_len = 27

    def __init__(self, y_size=4):
        super(BERT_Classifier2, self).__init__()
        # BERT用
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased')

    def forward(self, x_a, x_b):
        batch_size = len(x_a)
        tokens, token_types = self.encode(x_a, x_b)
        loss, scores = self.bert(tokens, token_type_ids=token_types)
        y = self.classifier(h.view(batch_size, -1))

        return y

    def encode(self, batch_a, batch_b):
        batch_a = [self.tokenizer.tokenize('[CLS]' + a.replace('_', ' ') + '[SEP]') for a in batch_a]
        batch_b = [self.tokenizer.tokenize(          b.replace('_', ' ') + '[SEP]') for b in batch_b]
        # BPEでトークンに分割
        batch_tokens = [a + b                   for a, b in zip(batch_a, batch_b)]
        batch_ttypes = [[0]*len(a) + [1]*len(b) for a, b in zip(batch_a, batch_b)]
        # 最大長にあわせてパディング
        batch_tokens = [tks + ['[PAD]'] * (self.max_seq_len - len(tks)) for tks in batch_tokens]
        batch_ttypes = [tps + [0]       * (self.max_seq_len - len(tps)) for tps in batch_ttypes]
        # トークンをidに
        batch_tokens = [self.tokenizer.convert_tokens_to_ids(tks)  for tks in batch_tokens]
        #
        batch_tokens = torch.LongTensor(batch_tokens).to(self.device)
        batch_ttypes = torch.LongTensor(batch_ttypes).to(self.device)

        return batch_tokens, batch_ttypes

    def to(self, device):
        self.device = device
        return self


def main():
    bert = BertForTokenClassification.from_pretrained('bert-base-uncased')
    bert.config.num_labels=4
    print(bert.config)

if __name__ == '__main__': main()
