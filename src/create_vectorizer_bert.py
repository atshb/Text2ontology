import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from pytorch_pretrained_bert import BertTokenizer, BertModel


#
def create_vectorizer(terms, device='cpu'):
    #
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model.to(device)

    tkns_list = [' '.join(['[CLS]', t, '[SEP]']) for t in terms]
    tkns_list = [tokenizer.tokenize(t) for t in tkns_list]

    seq_len = max(len(t) for t in terms)

    term2vec = dict()
    for i, (term, tkns) in enumerate(zip(terms, tkns_list)):
        tids = tokenizer.convert_tokens_to_ids(tkns) + [0] * (seq_len - len(tkns))
        mask = [1] * len(tkns)                       + [0] * (seq_len - len(tkns))

        tids = torch.tensor([tids]).to(device)
        mask = torch.tensor([mask]).to(device)

        feature, _ = model(tids, None, mask)
        feature = feature[-1].view(-1).detach().cpu().numpy()

        term2vec[term] = feature

        if i % 100 == 0: print(i, feature.shape, tkns)
    return term2vec






#
def main():
    with open('../../dataset/wordnet_full.pickle', 'rb') as f: dataset = pickle.load(f)

    term_set = set()
    for a, b, l in dataset:
        term_set.add(' '.join(a))
        term_set.add(' '.join(b))
    print(len(term_set))

    if torch.cuda.is_available(): device = 'cuda'
    else                        : device = 'cpu'

    term2vec = create_vectorizer(term_set, device)

    with open('bert_cat.pickle', 'wb') as f: pickle.dump(term2vec, f)


if __name__ == '__main__': main()
