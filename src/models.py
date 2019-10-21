import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


'''
'''
class TwinRnnClassifier(nn.Module):

    def __init__(self, x_size, h_size=128, y_size=4, num_cell=1, drop_rate=0.2):
        super(TwinRnnClassifier, self).__init__()
        # あとで使うパラメーター
        self.h_size   = h_size
        self.num_cell = num_cell
        # BiLSTM
        rnn_drop = 0 if num_cell == 1 else drop_rate # ドロップアウトは最終層以外に適応されるので一層の場合は必要なし。
        self.lstm_a = nn.LSTM(x_size, h_size, num_layers=num_cell, batch_first=True, dropout=rnn_drop, bidirectional=True)
        self.lstm_b = nn.LSTM(x_size, h_size, num_layers=num_cell, batch_first=True, dropout=rnn_drop, bidirectional=True)
        # MLP
        self.classifier = nn.Sequential(
            # bidirectional かつ 二つの入力なので hidden size は4倍
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(4*h_size, 4*h_size), nn.ReLU(inplace=True), nn.Dropout(),
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
        # RNNのレイヤー数が１でない場合は最終層の出力だけ利用
        if self.num_cell != 1:
            a = a.view(self.num_cell, 2, batch_size, self.h_size)[-1]
            b = b.view(self.num_cell, 2, batch_size, self.h_size)[-1]
        # 双方向RNNは出力が２つなので連結
        a = torch.cat([e for e in a], dim=1)
        b = torch.cat([e for e in b], dim=1)
        # 二つの出力を連結
        return torch.cat((a, b), dim=1)


'''
'''
class Word2vecRnnClassifier(nn.Module):

    def __init__(self, config):
        super(Word2vecRnnClassifier, self).__init__()
        #
        self.config = config
        #
        self.w2v   = pd.read_pickle('../data/word2vec/word2vec.pkl')
        self.vocab = pd.read_pickle('../data/wordnet/vocabulary.pkl')
        self.padding = np.zeros(300)
        #
        self.classifier = TwinRnnClassifier(300)

    def embed_batch(self, batch):
        seq_len = max(len(l.split()) for l in batch)
        #
        batch = np.stack([self.embed_phrase(p, seq_len) for p in batch])
        batch = torch.from_numpy(batch).float().to(self.config['--device'])
        return batch

    def embed_phrase(self, phrase, seq_len):
        words = phrase.split()
        vecs =  [self.w2v[w] if w in self.w2v else self.padding for w in words]
        vecs += [self.padding] * (seq_len - len(words))
        return np.stack(vecs)

    def forward(self, x_a, x_b):
        e_a = self.embed_batch(x_a)
        e_b = self.embed_batch(x_b)
        y = self.classifier(e_a, e_b)
        return y


'''
'''
class BertClassifier(nn.Module):

    def __init__(self, args):
        super(BertClassifier, self).__init__()
        # パラメーター
        self.args = args
        self.seq_len = 15
        # BERT用トークナイザー
        self.tokenizer = BertTokenizer.from_pretrained(args['--weights'])
        # BERTによる文章分類モデル
        config = BertConfig(num_labels=4)
        self.model = BertForSequenceClassification.from_pretrained(args['--weights'], config=config)

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
        batch_tokens = [tks + ['[PAD]'] * (self.seq_len - len(tks)) for tks in batch_tokens]
        batch_ttypes = [tps + [0]       * (self.seq_len - len(tps)) for tps in batch_ttypes]
        # トークンをidに
        batch_tokens = [self.tokenizer.convert_tokens_to_ids(tks) for tks in batch_tokens]
        #
        batch_tokens = torch.LongTensor(batch_tokens).to(self.args['--device'])
        batch_ttypes = torch.LongTensor(batch_ttypes).to(self.args['--device'])
        return batch_tokens, batch_ttypes




def test():
    pretrained_weights = 'bert-base-uncased'

    model = BertForTokenClassification.from_pretrained(pretrained_weights)
    model.config.num_labels=4


    y = self.model(tokens, token_type_ids=token_types)


if __name__ == '__main__': test()
