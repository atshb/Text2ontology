import pandas as pd
import torch.utils.data as data


'''
Wordnet_Dataset
'''
class Wordnet_dataset(data.Dataset):

    def __init__(self, mode='train', num_data=-1):
        if   mode == 'train': self.data = pd.read_pickle('../data/wordnet/train.pkl')[:num_data]
        elif mode == 'valid': self.data = pd.read_pickle('../data/wordnet/valid.pkl')[:num_data]
        elif mode == 'test' : self.data = pd.read_pickle('../data/wordnet/test.pkl' )[:num_data]
        else : raise Exception('please select mode from ("train" | "valid" | "test")')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
