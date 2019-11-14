'''
Extract Phrase Pairs and the Relationship Label from Wordnet and Save the Dataset.

Usage:
    preprocess_wordnet.py [--max_num_char=<mc>]
                          [--max_num_word=<mw>]
                          [--num_unrelated=<nu>]
                          [--num_valid=<nv>]
                          [--num_test=<nd>]
    preprocess_wordnet.py (-h | --help)

Options:
    -h --help             show this help message and exit.
    --max_num_char=<mc>   maximum number of char.    [default: 1000]
    --max_num_word=<mw>   maximum number of word.    [default: 1000]
    --num_unrelated=<nu>  number of unrelated data.  [default: 500000]
    --num_valid=<nv>      number of validation data. [default: 10000]
    --num_test=<nd>       number of test data.       [default: 10000]
'''

import random
import pandas as pd
from nltk.corpus import wordnet as wn
from docopt import docopt
from collections import Counter

# 出力結果が毎回同じになるようにがseed値は固定
random.seed(128)
dir = '../data/wordnet/'

# 引数の取得
args = docopt(__doc__)
print(args)
max_num_char  = int(args['--max_num_char'])
max_num_word  = int(args['--max_num_word'])
num_unrelated = int(args['--num_unrelated'])
num_valid     = int(args['--num_valid'])
num_test      = int(args['--num_test' ])


# Wordnetからのデータの抽出
## 全lemmaをから単語数がmax_seq_len以下のものだけ取り出す
lemmas = wn.all_lemma_names(pos='n')
## フィルタリング
lemma_set = set(l for l in lemmas if len(l) <= max_num_char and len(l.split('_')) <= max_num_word)

## 保存
pd.to_pickle(lemma_set, dir + 'vocabulary.pkl')

## 同義語のペアの追加
synonyms = []
for s in wn.all_synsets(pos='n'):
    for a in s.lemma_names():
        for b in s.lemma_names():
            if a in lemma_set and b in lemma_set and a != b:
                synonyms.append((a, b))
## 上位下位、下位上位のペアの追加
sup_subs = []
sub_sups = []
for s in wn.all_synsets(pos='n'):
    hypos = s.hyponyms()
    for h in hypos:
        for a in s.lemma_names():
            for b in h.lemma_names():
                if a in lemma_set and b in lemma_set:
                    sup_subs.append((a, b))
                    sub_sups.append((b, a))
## 無関係ペアの追加
unrelated = []
lemma_list = list(lemma_set)
related_set = set(synonyms + sup_subs + sub_sups)
while len(unrelated) < num_unrelated:
    a = random.choice(lemma_list)
    b = random.choice(lemma_list)
    if (a, b) not in related_set: unrelated.append((a, b))


# データセットの作成
## 抽出したデータのラベル付けと＿をスペースに置換
synonyms  = [(a.replace('_', ' '), b.replace('_', ' '), 0) for a, b in synonyms ]
sup_subs  = [(a.replace('_', ' '), b.replace('_', ' '), 1) for a, b in sup_subs ]
sub_sups  = [(a.replace('_', ' '), b.replace('_', ' '), 2) for a, b in sub_sups ]
unrelated = [(a.replace('_', ' '), b.replace('_', ' '), 3) for a, b in unrelated]

print('num of synonym   pairs', len(synonyms ))
print('num of super-sub pairs', len(sup_subs ))
print('num of sub-super pairs', len(sub_sups ))
print('num of unrelated pairs', len(unrelated))

## データの統合とシャッフル
dataset = synonyms + sup_subs + sub_sups + unrelated
random.shuffle(dataset)

## トレーニング用とテスト用に分割
num_train = len(dataset) - (num_valid + num_test)

train_dataset = dataset[:num_train]
valid_dataset = dataset[num_train:-num_test]
test_dataset  = dataset[-num_test:]

## データセットをpickle、csvで保存
pd.to_pickle(train_dataset, dir + 'train.pkl')
pd.to_pickle(valid_dataset, dir + 'valid.pkl')
pd.to_pickle(test_dataset , dir + 'test.pkl')

# columns = ('Lemma A', 'Lemma B', 'Label')
# pd.DataFrame(train_dataset, columns=columns).to_csv(fname_train + '.csv', index=None)
# pd.DataFrame(valid_dataset, columns=columns).to_csv(fname_valid + '.csv', index=None)

print('num of train data :', len(train_dataset))
print('num of valid data :', len(valid_dataset))
print('num of test  data :', len(test_dataset ))
