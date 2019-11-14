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
TwinPhraseEncoder
    BERTに二つの文を単一入力として渡すための transform クラス。
    各文は Byte Pair Encoding(BPE) によりサブワードに分割され、さらにIDに変換される。
    また、二つの文を区別するため、トークンタイプ行列とセットで出力。
'''
class TwinPhraseEncoder():

    def __init__(self, tokenizer, seq_len=50):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __call__(self, x):
        '''
        # BPEでトークンに分割
        a = self.tokenizer.tokenize('[CLS]' + x[0] + '[SEP]')
        b = self.tokenizer.tokenize(          x[1] + '[SEP]')
        # トークンを連結し、トークンタイプ行列を作成
        tokens = a + b
        ttypes = [0] * len(a) + [1] * len(b)
        # トークンをidに
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # テンソル化 & 最大長にあわせてパディング
        tokens = torch.LongTensor(tokens + [0] * (self.seq_len - len(tokens)))
        ttypes = torch.LongTensor(ttypes + [0] * (self.seq_len - len(ttypes)))

        return tokens, ttypes
        '''
        encoded = self.tokenizer.encode_plus(x[0], x[1], max_length=self.seq_len)
        #
        tokens = encoded['input_ids']
        ttypes = encoded['token_type_ids']
        # テンソル化 & 最大長にあわせてパディング
        tokens += [0] * (self.seq_len - len(tokens))
        ttypes += [0] * (self.seq_len - len(ttypes))

        return torch.LongTensor(tokens), torch.LongTensor(ttypes)



'''
テスト用
'''
def test():
    from transformers import BertTokenizer, RobertaTokenizer

    # train = data.DataLoader(WordnetDataset('train', num_data=4, transform=transform), batch_size=2)

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    transform = TwinPhraseEncoder(tokenizer, seq_len=15)

    encoded = transform(('I have a pen', 'I have an apple'),)
    print(encoded[0])
    print(encoded[1])


if __name__ == '__main__': test()
