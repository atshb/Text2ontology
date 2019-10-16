import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


'''
'''
class BertClassifier(nn.Module):

    def __init__(self, pretrained_weights, max_seq_len=30):
        super(BertClassifier, self).__init__()
        # パラメーター
        self.max_seq_len = max_seq_len
        #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # BERT用トークナイザー
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # BERTによる文章分類モデル
        config = BertConfig(num_labels=4)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_weights, config=config)

    def forward(self, x_a, x_b):
        tokens, token_types = self.encode(x_a, x_b)
        y = self.model(tokens, token_type_ids=token_types)[0]
        return y

    def encode(self, batch_a, batch_b):
        # BPEでトークンに分割
        batch_a = [self.tokenizer.tokenize('[CLS]' + a + '[SEP]') for a in batch_a]
        batch_b = [self.tokenizer.tokenize(          b + '[SEP]') for b in batch_b]
        #
        batch_tokens = [a + b                   for a, b in zip(batch_a, batch_b)]
        batch_ttypes = [[0]*len(a) + [1]*len(b) for a, b in zip(batch_a, batch_b)]
        # 最大長にあわせてパディング
        batch_tokens = [tks + ['[PAD]'] * (self.max_seq_len - len(tks)) for tks in batch_tokens]
        batch_ttypes = [tps + [0]       * (self.max_seq_len - len(tps)) for tps in batch_ttypes]
        # トークンをidに
        batch_tokens = [self.tokenizer.convert_tokens_to_ids(tks) for tks in batch_tokens]
        #
        batch_tokens = torch.LongTensor(batch_tokens).to(self.device)
        batch_ttypes = torch.LongTensor(batch_ttypes).to(self.device)
        # print()
        # print(batch_tokens)
        # print(batch_ttypes)

        return batch_tokens, batch_ttypes




def test():
    pretrained_weights = 'bert-base-uncased'

    model = BertForTokenClassification.from_pretrained(pretrained_weights)
    model.config.num_labels=4


    y = self.model(tokens, token_type_ids=token_types)


if __name__ == '__main__': test()
