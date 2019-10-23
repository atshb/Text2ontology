import pandas as pd
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

vocab = pd.read_pickle('../data/wordnet/vocabulary.pkl')

max_len = max(len(tokenizer.encode(p)) for p in vocab)
print(max_len)
