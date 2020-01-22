import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Here is the sentence I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)
words = marked_text.split()

# Print out the tokens.
print ('BERT:',tokenized_text)
print('Word2Vec:',words)
