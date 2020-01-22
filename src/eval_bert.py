import torch
import torch.nn as nn
import torch.nn.functional as func
import pandas as pd
from math import log
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from inflector import Inflector
from dataset import *
from transformers import BertTokenizer, BertConfig, BertModel

from models_rnn import *

'''
'''
def extract_concepts(text, n_concept, ngram=3):
    #
    N = 4379810
    swords = set(stopwords.words('english') + [',', '.'])
    sdf = pd.read_pickle('../data/ngram_sdf/sdf.pkl')
    #
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    words = [Inflector().singularize(w) for w in words]
    #
    phrases = [words[j-i:j] for i in range(1, ngram+10) for j in range(i, len(words))]
    phrases = [p for p in phrases if all(w not in swords for w in p)]
    phrases = [' '.join(p) for p in phrases]
    phrases = [p for p in phrases if p in sdf]
    #
    tf = Counter(phrases)
    #
    tfidf_phrases = []
    for p in set(phrases):
        tfidf = tf[p] * log(N / sdf[p])
        tfidf_phrases.append((p, tfidf))
    #
    tfidf_phrases.sort(reverse=True, key=lambda t: t[1])
    top_phrases = [p for (p, t) in tfidf_phrases[:n_concept]]

    return top_phrases


'''
Phrase Pair Relationship Classifier(PPRC)
    実際のオントロジー生成タスクでのBERTベースモデルの性能評価用。
    モデルの出力にソフトマックスをかけて各ラベルの確率を戻す。
'''
class PPRC():
    def __init__(self, path, seq_len=30):

        # モデルの設定
        vec_size = 768
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        self.embed = BertModel.from_pretrained(pretrained_weights)
        self.embed.to('cpu').eval()

        self.model = SeriesClassifier(768,30)
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
        self.encoder = SeparatePhraseEncoder(tokenizer, seq_len)

    def __call__(self, a, b):
        #
        a, b = self.encoder((a, b))
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        with torch.no_grad():
            e_a = self.embed(a)[0]
            e_b = self.embed(b)[0]
        #
        y = self.model(e_a, e_b)
        y = func.softmax(y, dim=1)
        y = y.data.numpy().tolist()[0]
        y = [f'{p:>6.2%}' for p in y]
        #
        return y


'''
'''
def main():
    #
    #with open('../data/ontology/example3.txt') as f: text = f.read()
    #
    phrases = ['motor vehicle','car','automobile','vehicle']
    #
    pprc = PPRC('../result/bert.pkl', 30)
    results = [[a, b] + pprc(a, b) for b in phrases for a in phrases]
    #
    columns = ['phrase A', 'phrase B',
               'synonym', 'super-sub', 'sub-super', 'unrelated']
    results = pd.DataFrame(results, columns=columns)
    results.to_csv('../result/predict3.csv')


if __name__ == '__main__': main()
