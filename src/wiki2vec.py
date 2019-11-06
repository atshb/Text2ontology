import numpy as np
from nltk import tokenize
from wikipedia2vec import Wikipedia2Vec
import pickle
import pandas as pd

def get_wikipedia_vecs( vocab ):
    tokens, vecs = [], []

    wiki2vec = Wikipedia2Vec.load('enwiki_20180420_500d.pkl')

    for token in tokenize.word_tokenize(vocab):

        key = wiki2vec.get_entity(token)
        if key != None:
            tokens.append(key)
            vec = np.array(wiki2vec.get_entity_vector(token))
            vecs.append(vec)
            continue

        key = wiki2vec.get_word(token)
        if key != None:
            tokens.append(key)
            vec = np.array(wiki2vec.get_word_vector(token))
            vecs.append(vec)
            continue

    return tokens, vecs


def main():
    vocab = pd.read_pickle('../data/wordnet/vocabulary.pkl')
    tokens, vecs = get_wikipedia_vecs(vocab)

    print("vecs:",vecs[0].shape, ",tokens:",type(tokens[0]))
    '''for t, v in zip(tokens, vecs): print(f'{t.__str__():<20} {v[:5]}')'''


if __name__ == '__main__': main()
