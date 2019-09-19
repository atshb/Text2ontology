import random
import pandas as pd
from nltk.corpus import wordnet as wn


# 出力結果が毎回同じになるようにがseed値は固定
random.seed(1000)


# Parameters
max_seq_len = 3
n_unrelated = 500000
train_rate = 0.8

dir = '../data/wordnet/'
fname_train = dir + 'train.pkl'
fname_valid = dir + 'valid.pkl'


# Wordnetからのデータの抽出

## 全lemmaをから単語数がmax_seq_len以下のものだけ取り出す
lemmas = wn.all_lemma_names(pos='n')
lemma_set = set(l for l in lemmas if len(l.split('_')) <= max_seq_len)

## 同義語のペアの追加
synonyms = []
for s in wn.all_synsets(pos='n'):
    for a in s.lemma_names():
        for b in s.lemma_names():
            if a in lemma_set and b in lemma_set:
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
while len(unrelated) < n_unrelated:
    a = random.choice(lemma_list)
    b = random.choice(lemma_list)
    if (a, b) not in related_set: unrelated.append((a, b))


# データセットの作成

## 抽出したデータのラベル付け
synonyms  = [(a, b, 0) for a, b in synonyms ]
sup_subs  = [(a, b, 1) for a, b in sup_subs ]
sub_sups  = [(a, b, 2) for a, b in sub_sups ]
unrelated = [(a, b, 3) for a, b in unrelated]

## データの統合とシャッフル
dataset = synonyms + sup_subs + sub_sups + unrelated
random.shuffle(dataset)

## トレーニング用とテスト用に分割
n_train = round(len(dataset) * train_rate)
n_valid = len(dataset) - n_train
train_dataset = dataset[:n_train]
valid_dataset = dataset[n_train:]

## データセットをpickle、csvで保存
pd.to_pickle(train_dataset, fname_train)
pd.to_pickle(valid_dataset, fname_valid)
# columns = ('Lemma A', 'Lemma B', 'Label')
# pd.DataFrame(train_dataset, columns=columns).to_csv(fname_train + '.csv', index=None)
# pd.DataFrame(valid_dataset, columns=columns).to_csv(fname_valid + '.csv', index=None)
print('num of train data :', n_train)
print('num of valid data :', n_valid)
