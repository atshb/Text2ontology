import pickle
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from model import *



# param
max_epoch  = 50
batch_size = 32

# Loading
train = pd.read_pickle('../data/wordnet/train.pkl')
valid = pd.read_pickle('../data/wordnet/valid.pkl')

train_loader = data.DataLoader(Ont_Dataset(train), batch_size, shuffle=False)
valid_loader = data.DataLoader(Ont_Dataset(valid), batch_size, shuffle=False)

#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# embed = Word2vec_Embedding().to(device)
embed = BERT_Embedding().to(device)
# embed = XLNet_Embedding().to(device)
model = RNN_ONT(x_size=768).to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#
for epoch in range(max_epoch):

    # Training
    model.train()
    epoch_loss = 0
    for i, (x_a, x_b, t) in enumerate(train_loader):
        #
        e_a = embed(x_a)
        e_b = embed(x_b)
        t = t.to(device)
        # calculate loss
        y = model(e_a, e_b)
        loss = loss_func(y, t)
        epoch_loss += loss.cpu().item()
        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0: print(f'{i + 1:>4} : {loss.cpu().item():>6.3}')

    # Validation
    model.eval()
    epoch_accu = 0
    for x_a, x_b, t in valid_loader:
        #
        e_a = embed(x_a)
        e_b = embed(x_b)
        t = t.to(device)
        # calculate accuracy
        y = model(e_a, e_b)
        _, y = torch.max(y.data, 1)
        epoch_accu += sum(1 for y_i, t_i in zip(y, t) if y_i == t_i)

    # Show Progress
    epoch_loss /= len(train)
    epoch_accu /= len(valid)
    print(f'{epoch:0>2} | loss : {epoch_loss:>7.5f} | accu : {epoch_accu:.2%}')

    # Save Model
    torch.save(model.state_dict(), f'../result/bert-rnn/{epoch+1:0>2}.pkl')
