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
batch_size = 1024

# Loading
train = pd.read_pickle('../data/wordnet/train.pkl')
valid = pd.read_pickle('../data/wordnet/valid.pkl')

train_loader = data.DataLoader(Ont_Dataset(train), batch_size, shuffle=False)
valid_loader = data.DataLoader(Ont_Dataset(valid), batch_size, shuffle=False)

#
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embed = Word2vec_Embedding()
model = RNN_ONT().to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#
for epoch in range(max_epoch):
    # Training
    epoch_loss = 0
    model.train()
    for i, (x_a, x_b, t) in enumerate(train_loader):
        #
        e_a = embed(x_a)
        e_b = embed(x_b)
        #
        e_a = e_a.to(device)
        e_b = e_b.to(device)
        t = t.to(device)
        #
        y = model(e_a, e_b)
        loss = loss_func(y, t)
        epoch_loss += loss.cpu().item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i % 10 == 0: print(f'{i + 1:>4} : {loss.cpu().item():>6.3}')

    # Validation
    epoch_accu = 0
    model.eval()
    for a, b, t in valid_loader:
        #
        e_a = embed(x_a)
        e_b = embed(x_b)
        #
        e_a = e_a.to(device)
        e_b = e_b.to(device)
        t = t.to(device)
        #
        y = model(a, b)
        _, y = torch.max(y.data, 1)
        epoch_accu += sum(1 for y_i, t_i in zip(y, t) if y_i == t_i)


    # Show Progress
    epoch_loss /= n_train
    epoch_accu /= n_valid
    print(f'{epoch:0>2} | loss : {epoch_loss:>7.5f} | accu : {epoch_accu:.2%}')


if __name__ == '__main__': main()
