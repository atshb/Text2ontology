import torch
import torch.utils.data as data
import numpy  as np
import pandas as pd
from wikipedia2vec import Wikipedia2Vec

'''
WordnetDataset
    Wordnetから取得したデータを扱うクラス。
    data   : 語句のペア
    labels : 関係性ラベル（同義語、上位下位、下位上位、無関係）
'''
class WordnetDataset(data.Dataset):

    def __init__(self, mode='train', num_data=-1, transform=None):
        # 読み込み
        if   mode == 'train': data = pd.read_pickle('../data/wordnet/train.pkl')
        elif mode == 'valid': data = pd.read_pickle('../data/wordnet/valid.pkl')
        elif mode == 'test' : data = pd.read_pickle('../data/wordnet/test.pkl' )
        #
        self.data   = [(a, b) for a, b, _ in data[:num_data]]
        self.labels = [l      for _, _, l in data[:num_data]]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform: x = self.transform(x)
        return x, y


'''
Word2vecEmbedding
    語句のペアを学習済 Word Embedding でベクトル化する transform クラス。
    Dataset クラスに tranform クラスとして渡される。
    単語数が seq_len 未満の場合はゼロベクトルでパディング。
    in  : 語句のペア
    out : 行列 (seq_len, 300) のペア
'''
class Word2vecEmbedding():

    def __init__(self, seq_len=20):
        self.w2v = pd.read_pickle('../data/word2vec/word2vec.pkl')
        self.seq_len = seq_len
        self.padding = np.zeros(300)

    def __call__(self, x):
        return self.embed_phrase(x[0]), self.embed_phrase(x[1])

    def embed_phrase(self, phrase):
        words = phrase.split()
        vecs =  [self.w2v[w] if w in self.w2v else self.padding for w in words]
        vecs += [self.padding] * (self.seq_len - len(words))
        return np.stack(vecs).astype(np.float32)


'''
Wiki2vecEmbedding
    語句のペアを学習済 Word Embedding でベクトル化する transform クラス。
    Dataset クラスに tranform クラスとして渡される。
    単語数が seq_len 未満の場合はゼロベクトルでパディング。
    in  : 語句のペア
    out : 行列 (seq_len, 300) のペア
'''
class Wiki2vecEmbedding():

    def __init__(self, seq_len=20, use_entity=False):
        self.wiki2vec = Wikipedia2Vec.load('../data/wiki2vec/enwiki_300d.pkl')
        self.seq_len = seq_len
        self.use_entity = use_entity
        self.padding = np.zeros(300)

    def __call__(self, x):
        return self.embed_phrase(x[0]), self.embed_phrase(x[1])

    def embed_phrase(self, phrase):
        entity = self.wiki2vec.get_entity(phrase.capitalize())
        if entity and self.use_entity:
            vecs = [self.wiki2vec.get_entity_vector(entity.title)]
            vecs += [self.padding] * (self.seq_len - 1)
        else:
            words = phrase.split()
            words = [self.wiki2vec.get_word(w) for w in words]
            words = [w.text if w else w for w in words]
            vecs =  [self.wiki2vec.get_word_vector(w) if w else self.padding for w in words]
            vecs += [self.padding] * (self.seq_len - len(words))

        return np.stack(vecs).astype(np.float32)


'''
ConcatPhraseEncoder
    BERTに二つの文を単一入力として渡すための transform クラス。
    各文は Byte Pair Encoding(BPE) によりサブワードに分割され、さらにIDに変換される。
    また、二つの文を区別するため、トークンタイプ行列とセットで出力。
'''
class ConcatPhraseEncoder():

    def __init__(self, tokenizer, seq_len=50):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __call__(self, x):
        # BPEでトークンに分割
        a = self.tokenizer.tokenize('[CLS]' + x[0] + '[SEP]')
        b = self.tokenizer.tokenize(          x[1] + '[SEP]')
        # トークンを連結し、トークンタイプ行列を作成
        tokens = a + b
        ttypes = [0] * len(a) + [1] * len(b)
        # トークンをidに
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # テンソル化 & 最大長にあわせてパディング
        tokens += [0] * (self.seq_len - len(tokens))
        ttypes += [0] * (self.seq_len - len(ttypes))

        return torch.LongTensor(tokens), torch.LongTensor(ttypes)


'''
SeparatePhraseEncoder
    BERTに二つの文を別々の入力として渡すための transform クラス。
    各文は Byte Pair Encoding(BPE) によりサブワードに分割され、さらにIDに変換される。
    出力は分割された文章とIDのリストになった入力文Aと入力文B。
'''
class SeparatePhraseEncoder():

    def __init__(self, tokenizer, seq_len=25, with_phrase=False):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.with_phrase = with_phrase

    def __call__(self, x):
        # BPEでトークンに分割
        a = self.tokenizer.tokenize('[CLS]' + x[0] + '[SEP]')
        b = self.tokenizer.tokenize('[CLS]' + x[1] + '[SEP]')
        # トークンをidに
        a_id = self.tokenizer.convert_tokens_to_ids(a)
        b_id = self.tokenizer.convert_tokens_to_ids(b)
        # テンソル化 & 最大長にあわせてパディング
        a += ['[PAD]'] * (self.seq_len - len(a))
        b += ['[PAD]'] * (self.seq_len - len(b))
        a_id += [0] * (self.seq_len - len(a_id))
        b_id += [0] * (self.seq_len - len(b_id))
        #
        a_id = torch.LongTensor(a_id)
        b_id = torch.LongTensor(b_id)

        if self.with_phrase: return a_id, b_id, a, b
        else               : return a_id, b_id


'''
テスト用
'''
def test():
    pass


if __name__ == '__main__': test()
