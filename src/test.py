'''
Train model for classification of relationship between Compound words

Usage:
    train.py (word2vec | bert | xlnet) [--num_train=<nt>] [--num_valid=<nv>] [--batch_size=<bs>] [--max_epoch=<me>]
    train.py -h | --help

Options:
    -h --help          show this help message and exit.
    --num_train=<nt>   number of training   data [default: -1].
    --num_valid=<nv>   number of validation data [default: -1].
    --batch_size=<bs>  size of batch [default: 10].
    --max_epoch=<me>   maximum training epoch.
'''

from docopt import docopt

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
