import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
sys.path.append('..')
from utils.misc import *
from utils.data_process import *



def batch_generator(data):
    collate_fn = PaddingCollate()
    return collate_fn([data])


def one_hot(x, bins):
    if x.size()[-1] != 1:
        x = x.unsqueeze(-1)
    if len(x.size()) == 3:
        i, j, k = x.size()
        p = torch.full((i, j, len(bins)), fill_value=0).to(x)
        b = torch.argmin(torch.abs(x - (torch.tensor(bins).to(x))[None, None, :]), dim=2)
        p = p.scatter_(2, b.unsqueeze(2), 1)
    elif len(x.size()) == 4:
        i, j, k,u = x.size()
        p = torch.full((i, j, k, len(bins)), fill_value=0).to(x)
        b = torch.argmin(torch.abs(x - (torch.tensor(bins).to(x))[None, None, :]), dim=3)
        p = p.scatter_(3, b.unsqueeze(3), 1)
    return p


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


def load_weight(weight):
    import collections
    tem_OrderedDict = collections.OrderedDict()
    for k, v in weight.items():
        if k.split('.', 1)[0] != '_orig_mod': return weight
        tem_k = k.split('.', 1)[1]
        tem_OrderedDict[tem_k] = v
    return tem_OrderedDict