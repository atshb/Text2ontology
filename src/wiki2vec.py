import numpy as np
import wikipedia2vec
from wikipedia2vec import Wikipedia2Vec



def get_wikipedia_vecs(phrase, wiki2vec):
    entity = phrase.capitalize()
    key = wiki2vec.get_entity(entity)
    if key:
        print('ue')
        vecs  = [wiki2vec.get_entity_vector(entity)]
        words = [entity]

    else:
        print('sita')
        words = phrase.split()
        words = [wiki2vec.get_word(w) for w in words]
        words = [w.text if w else w for w in words]
        vecs = [wiki2vec.get_word_vector(w) if w else np.zeros(100) for w in words]

    return words, np.stack(vecs)


def main():
    wiki2vec = Wikipedia2Vec.load('../data/wiki2vec/enwiki_100d.pkl')
    words, vecs = get_wikipedia_vecs('i have a pen .', wiki2vec)

    print("vecs:",vecs.shape, ",phrases:",words)


if __name__ == '__main__': main()
