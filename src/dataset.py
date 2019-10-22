import torch
import torch.utils.data as data
import numpy  as np
import pandas as pd


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
TwinPhraseEncoder
    BERTに二つの文を単一入力として渡すための transform クラス。
    各文は Byte Pair Encoding(BPE) によりサブワードに分割され、さらにIDに変換される。
    また、二つの文を区別するため、トークンタイプ行列とセットで出力。
'''
class TwinPhraseEncoder():

    def __init__(self, tokenizer, seq_len=70):
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
        tokens = torch.LongTensor(tokens + [0] * (self.seq_len - len(tokens)))
        ttypes = torch.LongTensor(ttypes + [0] * (self.seq_len - len(ttypes)))

        return tokens, ttypes


'''
テスト用
'''
def test():
    # transform = TwinPhraseEncoder('bert-base-uncased')
    # train = data.DataLoader(WordnetDataset('train', num_data=4, transform=transform), batch_size=2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encoded = tokenizer.encode_plus('I have a pen', 'I have an apple', max_length=30)
    token_ids = encoded['input_ids']
    ttype_ids = encoded['token_type_ids']
    token_ids += [0] * (seq_len - len(token_ids))
    ttype_ids += [0] * (seq_len - len(ttype_ids))


if __name__ == '__main__': test()
