'''
Train model for classification of relationship between Compound words

Usage:
    test_bert_rnn.py (-h | --help)
    test_bert_rnn.py (rnn | cnn | series | parallel)
                     [--model_path=<mp>]
                     [--seq_len=<sl>]
                     [--batch_size=<bs>]
                     [--num_test=<nt>]

Options:
    -h --help          show this help message and exit.
    --model_path=<mp>  path of model.             [default: ../result/bert.pkl]
    --seq_len=<sl>     maximum sequence length.   [default: 30]
    --batch_size=<bs>  size of mini-batch.        [default: 64]
    --num_test=<nt>    number of test data.       [default: -1]
'''

import torch
import torch.nn as nn
import torch.optim as optim

from docopt import docopt
from pprint import pprint
from tqdm import tqdm
from dataset import *
from models_rnn import *

'''
'''
def test_model(model, embed, dataloader, device):
    model.eval()

    corrects   = []
    incorrects = []
    with torch.no_grad():
        for (x_a, x_b, a, b), t in dataloader:
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            td = t.to(device)
            # calculate loss
            with torch.no_grad():
                e_a = embed(x_a)[0]
                e_b = embed(x_b)[0]
            y = model(e_a, e_b)
            _, p = torch.max(y, 1)
            #
            # epoch_accu += torch.sum(p == t).item()

            # print(a)
            # print(b)
            for i in range(len(p)):
                a_i = ' '.join(s[i] for s in a if s[i] not in ['[PAD]'])
                b_i = ' '.join(s[i] for s in b if s[i] not in ['[PAD]'])
                if p[i] == t[i]:   corrects.append((a_i, b_i, t[i].item(), p[i].item()))
                else           : incorrects.append((a_i, b_i, t[i].item(), p[i].item()))

    return corrects, incorrects


'''
'''
def main():
    # コマンドライン引数の取得（このファイル上部のドキュメントから自動生成）
    args = docopt(__doc__)
    pprint(args)

    # パラメータの取得
    model_path = args['--model_path']
    seq_len    = int(args['--seq_len'])
    batch_size = int(args['--batch_size'])
    num_test   = int(args['--num_test'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルの選択
    vec_size  = 300
    embedding = Word2vecEmbedding(seq_len)

    # 学習モデル
    if   args['rnn']     : model = RnnClassifier(vec_size)
    elif args['cnn']     : model = CnnClassifier(vec_size, seq_len)
    elif args['series']  : model = SeriesClassifier(vec_size, seq_len)
    elif args['parallel']: model = ParallelClassifier(vec_size, seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # データの読み込みとデータセットの作成

    test_dataset = WordnetDataset(mode='test', num_data=num_test, transform=encoder)
    test_loader = data.DataLoader(test_dataset, batch_size, shuffle=False)

    corrects, incorrects = test_model(model, embedding, test_loader, device)

    rel_dict = {0:'synonym', 1:'hypernym', 2:'hyponym', 3:'unrelated'}
    print('='*20 + ' correct ' + '='*20)
    for i, (a, b, t, p) in enumerate(incorrects):
        print(i+1)
        print(f'A : {a}')
        print(f'B : {b}')
        print(f'o : {rel_dict[t]:<9}')
        print('-'*49)
    print('='*20 + ' incorrect ' + '='*20)
    for i, (a, b, t, p) in enumerate(incorrects):
        print(i+1)
        print(f'A : {a}')
        print(f'B : {b}')
        print(f'o : {rel_dict[t]:<9}')
        print(f'x : {rel_dict[p]:<9}')
        print('-'*51)

    pd.to_pickle(  corrects, '../result/word2vec_rnn_correct.pkl')
    pd.to_pickle(incorrects, '../result/word2vec_rnn_incorrect.pkl')
if __name__ == '__main__': main()
